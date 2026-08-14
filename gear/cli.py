"""Command-line interface for ASPR-GEAR review, validation, and batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .calibration import CalibrationService
from .calibration_assets import (
    RELEASE_ALIAS,
    load_calibration_release,
    promote_calibration_release,
    sha256_file,
)
from .config import PROJECT_ROOT, AssetPaths, load_config
from .contracts import PaperMetadata, ReviewRequest
from .review_contracts import ReviewBundle, VerificationIssue
from .review_pipeline import review_paper
from .trace import EvidenceStore
from .review_verifier import ReviewVerifier


def _metadata(path: Optional[Path]) -> PaperMetadata:
    if path is None:
        return PaperMetadata()
    return PaperMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def _review_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    request = ReviewRequest(
        paper_path=args.paper,
        metadata=_metadata(args.metadata),
    )
    bundle = review_paper(
        request,
        output_dir=args.output_dir,
        config=config,
    )
    print(
        json.dumps(
            {"status": bundle.status.value, **bundle.output_files},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if bundle.status.value != "failed" else 2


def _validate_command(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    raw = (run_dir / "review_bundle.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    store = EvidenceStore(run_dir)
    bundle = ReviewBundle.model_validate(payload)
    report = ReviewVerifier().verify(
        bundle.structured_review, bundle.state, bundle.paper_ir, store
    )
    manifest_failures = store.validate_manifest()
    if manifest_failures:
        manifest_issues = [
            VerificationIssue(
                issue_id="VI-"
                + hashlib.sha256(reason.encode("utf-8")).hexdigest()[:18],
                code="run_manifest_integrity",
                message=reason,
                repairable=False,
            )
            for reason in manifest_failures
        ]
        report = report.model_copy(
            update={"passed": False, "issues": [*report.issues, *manifest_issues]}
        )
    (run_dir / "revalidation_report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    print(report.model_dump_json(indent=2))
    return 0 if report.passed else 1


def _benchmark_command(args: argparse.Namespace) -> int:
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = payload.get("cases", []) if isinstance(payload, dict) else []
    results: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        metadata = PaperMetadata.model_validate(case.get("metadata") or {})
        request = ReviewRequest(
            paper_path=Path(case["paper_path"]),
            metadata=metadata,
        )
        output = args.output_dir / str(case.get("case_id") or f"case_{index:04d}")
        bundle = review_paper(
            request, output_dir=output, config=load_config(args.config)
        )
        results.append(
            {
                "case_id": case.get("case_id", index),
                "status": bundle.status.value,
                "verification_passed": bundle.verification.passed,
                "output_dir": str(output),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark_results.json").write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if results and all(item["verification_passed"] for item in results) else 1


def _validate_assets_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = CalibrationService(config).validate_official_replay(
        batch_size=args.batch_size
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def _promotion_sources(path: Path, config: Any) -> Dict[str, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") != "aspr_calibration_promotion_source_v1":
        raise ValueError("unsupported calibration promotion source contract")
    if payload.get("alias") != RELEASE_ALIAS:
        raise ValueError("promotion source targets an unsupported calibration alias")
    sources = {
        name: config.resolve_path(value)
        for name, value in (payload.get("assets") or {}).items()
    }
    for source in sources.values():
        config.validate_asset_path(source)
    return sources


def _promote_calibration_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    sources = _promotion_sources(args.source_config.resolve(), config)
    core_paths = AssetPaths.model_validate(
        {name: sources[name] for name in AssetPaths.model_fields}
    )
    replay = CalibrationService(
        config, asset_paths=core_paths
    ).validate_official_replay(batch_size=args.batch_size)
    model_hash = sha256_file(core_paths.official_model_joblib).split(":", 1)[1][:8]
    matrix_hash = sha256_file(core_paths.feature_matrix_16).split(":", 1)[1][:8]
    release_id = args.release_id or (f"pgc-v3-d5-fulltext16-{model_hash}-{matrix_hash}")
    release = promote_calibration_release(
        release_id=release_id,
        source_assets=sources,
        replay=replay,
        source_manifest_sha256=sha256_file(core_paths.official_run_manifest),
        registry_path=config.resolve_path(config.calibration_registry),
    )
    print(
        json.dumps(
            {
                "release_id": release.release_id,
                "asset_root": str(release.asset_root),
                "manifest": str(release.manifest_path),
                "replay": replay,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _show_calibration_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    release = load_calibration_release(
        args.release or config.calibration_release,
        registry_path=config.resolve_path(config.calibration_registry),
        verify=args.verify,
    )
    print(
        json.dumps(
            {
                "release_id": release.release_id,
                "asset_root": str(release.asset_root),
                "manifest": str(release.manifest_path),
                "verified": bool(args.verify),
                "assets": {
                    name: str(release.path(name))
                    for name in sorted(release.manifest.assets)
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASPR-GEAR evidence-adaptive reviewer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    review = subparsers.add_parser("review", help="Review one Markdown or PDF paper")
    review.add_argument(
        "--paper",
        "--pdf",
        dest="paper",
        type=Path,
        required=True,
        help="Markdown manuscript (.md/.markdown) or PDF ingress adapter (.pdf)",
    )
    review.add_argument("--metadata", type=Path)
    review.add_argument("--output-dir", type=Path)
    review.add_argument("--config", type=Path)
    review.set_defaults(handler=_review_command)
    validate = subparsers.add_parser("validate-run", help="Revalidate an existing run")
    validate.add_argument("run_dir", type=Path)
    validate.set_defaults(handler=_validate_command)
    benchmark = subparsers.add_parser("benchmark", help="Run a JSON benchmark manifest")
    benchmark.add_argument("--manifest", type=Path, required=True)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--config", type=Path)
    benchmark.set_defaults(handler=_benchmark_command)
    assets = subparsers.add_parser(
        "validate-assets",
        help="Replay all frozen Fig.3 rows through the official model",
    )
    assets.add_argument("--config", type=Path)
    assets.add_argument("--batch-size", type=int, default=50_000)
    assets.set_defaults(handler=_validate_assets_command)
    promote = subparsers.add_parser(
        "promote-calibration",
        help="Validate and atomically publish a frozen calibration release",
    )
    promote.add_argument(
        "--source-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "gear" / "calibration_promotion_source.json",
    )
    promote.add_argument("--release-id")
    promote.add_argument("--batch-size", type=int, default=50_000)
    promote.add_argument("--config", type=Path)
    promote.set_defaults(handler=_promote_calibration_command)
    show = subparsers.add_parser(
        "show-calibration",
        help="Resolve the active calibration release and list local assets",
    )
    show.add_argument("--release")
    show.add_argument("--verify", action="store_true")
    show.add_argument("--config", type=Path)
    show.set_defaults(handler=_show_calibration_command)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


__all__ = ["build_parser", "main"]

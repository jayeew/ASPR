"""Command-line interface for ASPR-GEAR review, validation, and batches."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .config import load_config
from .contracts import PaperMetadata, ReviewRequest
from .diffusion_forecast import validate_runtime_replay
from .review_contracts import ReviewBundle, VerificationIssue
from .review_pipeline import review_paper
from .review_verifier import ReviewVerifier
from .trace import EvidenceStore


def _metadata(path: Path | None) -> PaperMetadata:
    if path is None:
        return PaperMetadata()
    return PaperMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _review_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    request = ReviewRequest(
        paper_path=args.paper,
        metadata=_metadata(args.metadata),
        evaluation_date=(
            date.fromisoformat(args.cutoff)
            if args.cutoff
            else datetime.now().astimezone().date()
        ),
    )
    bundle = review_paper(
        request,
        output_dir=args.output_dir,
        config=config,
        full_artifacts=not args.compact_artifacts,
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
    if payload.get("schema_revision") != "evidence_state_delta_v2":
        print(
            json.dumps(
                {
                    "passed": False,
                    "code": "unsupported_schema_revision",
                    "message": "This run predates the Evidence-State Delta contract; rerun it.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    store = EvidenceStore(run_dir)
    bundle = ReviewBundle.model_validate(payload)
    verifier = ReviewVerifier()
    if bundle.state is not None:
        report = verifier.verify_state(
            bundle.structured_review, bundle.state, bundle.paper_ir, store
        )
    else:
        raise ValueError("run bundle has no review state")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    indexed = list(enumerate(cases))

    def worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        return _run_benchmark_case(item[0], item[1], args)

    if args.workers == 1:
        results = [worker(item) for item in indexed]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(worker, indexed))
    results.sort(key=lambda row: row["index"])
    for row in results:
        row.pop("index", None)
    (args.output_dir / "batch_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {"case_count": len(results), "status_counts": counts, "results": results}
    (args.output_dir / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    failed = any(row["status"] == "failed" for row in results)
    return 1 if args.strict_exit and failed else 0


def _run_benchmark_case(
    index: int, case: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    started = time.monotonic()
    case_id = str(case.get("case_id") or f"case_{index:04d}")
    output = (args.output_dir / case_id).resolve()
    base = {
        "index": index,
        "case_id": case_id,
        "paper_id": str(case.get("paper_id") or ""),
        "output_dir": str(output),
    }
    existing = _existing_case_result(output)
    if (
        args.resume
        and existing is not None
        and not (args.retry_failed and existing["status"] == "failed")
    ):
        return {**base, **existing, "elapsed_seconds": 0.0, "resumed": True}
    try:
        paper = Path(str(case["paper_path"])).resolve()
        if not paper.is_file():
            raise FileNotFoundError(paper)
        metadata = PaperMetadata.model_validate(case.get("metadata") or {})
        output.mkdir(parents=True, exist_ok=True)
        metadata_path = output / "batch_metadata.json"
        metadata_path.write_text(
            metadata.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        command = [
            sys.executable,
            "-m",
            "gear",
            "review",
            "--paper",
            str(paper),
            "--metadata",
            str(metadata_path),
            "--output-dir",
            str(output),
        ]
        if case.get("cutoff"):
            command.extend(["--cutoff", str(case["cutoff"])])
        if args.config:
            command.extend(["--config", str(args.config.resolve())])
        if not args.full_artifacts:
            command.append("--compact-artifacts")
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=args.case_timeout_seconds,
        )
        (output / "subprocess_stdout.log").write_text(
            completed.stdout or "", encoding="utf-8"
        )
        (output / "subprocess_stderr.log").write_text(
            completed.stderr or "", encoding="utf-8"
        )
        metadata_path.unlink(missing_ok=True)
        existing = _existing_case_result(output)
        if existing is None:
            reason = f"subprocess_exit_{completed.returncode}"
            existing = {"status": "failed", "reason_codes": [reason]}
        return {
            **base,
            **existing,
            "elapsed_seconds": time.monotonic() - started,
            "resumed": False,
        }
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        subprocess.TimeoutExpired,
        TypeError,
        ValueError,
    ) as exc:
        return {
            **base,
            "status": "failed",
            "reason_codes": [f"{type(exc).__name__}:{exc}"],
            "elapsed_seconds": time.monotonic() - started,
            "resumed": False,
        }


def _existing_case_result(output: Path) -> dict[str, Any] | None:
    path = output / "review_bundle.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        status = str(payload["status"])
        reasons = [
            str(item.get("reason"))
            for item in (payload.get("state") or {}).get("failures", [])
            if item.get("reason")
        ]
        reasons.extend(
            str(reason)
            for reason in (payload.get("process_diagnostic") or {}).get(
                "blocking_reasons", []
            )
            if reason
        )
        return {"status": status, "reason_codes": list(dict.fromkeys(reasons))}
    except (json.JSONDecodeError, KeyError, OSError, TypeError):
        return None


def _validate_assets_command(args: argparse.Namespace) -> int:
    config = load_config(args.config, validate_assets=True)
    report = validate_runtime_replay(config.resolved_forecast_release_manifest())
    anatomy_manifest = config.resolved_forecast_anatomy_manifest()
    if anatomy_manifest is not None:
        anatomy = json.loads(anatomy_manifest.read_text(encoding="utf-8"))
        report["forecast_anatomy_release_id"] = anatomy.get("release_id")
        report["forecast_anatomy_rows"] = anatomy.get("row_count")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


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
    review.add_argument(
        "--cutoff",
        help="Evidence cutoff date in YYYY-MM-DD form; metadata dates take precedence.",
    )
    review.add_argument("--output-dir", type=Path)
    review.add_argument("--config", type=Path)
    review.add_argument(
        "--compact-artifacts", action="store_true", help=argparse.SUPPRESS
    )
    review.set_defaults(handler=_review_command)
    validate = subparsers.add_parser("validate-run", help="Revalidate an existing run")
    validate.add_argument("run_dir", type=Path)
    validate.set_defaults(handler=_validate_command)
    benchmark = subparsers.add_parser("benchmark", help="Run a JSON benchmark manifest")
    benchmark.add_argument("--manifest", type=Path, required=True)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--config", type=Path)
    benchmark.add_argument("--workers", type=_positive_int, default=1)
    benchmark.add_argument("--resume", action="store_true")
    benchmark.add_argument("--retry-failed", action="store_true")
    benchmark.add_argument("--strict-exit", action="store_true")
    benchmark.add_argument("--full-artifacts", action="store_true")
    benchmark.add_argument("--case-timeout-seconds", type=_positive_int, default=7200)
    benchmark.set_defaults(handler=_benchmark_command)
    assets = subparsers.add_parser(
        "validate-assets",
        help="Replay all frozen Fig.3 rows through the official model",
    )
    assets.add_argument("--config", type=Path)
    assets.set_defaults(handler=_validate_assets_command)
    modules = subparsers.add_parser(
        "module", help="Publish, resolve, or compare decoupled module artifacts"
    )
    modules.add_argument("module_args", nargs=argparse.REMAINDER)
    modules.set_defaults(handler=_module_command)
    return parser


def _module_command(args: argparse.Namespace) -> int:
    from .module_cli import main as module_main

    return module_main(args.module_args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


__all__ = ["build_parser", "main"]

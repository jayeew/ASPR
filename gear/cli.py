"""Command-line interface for ASPR-GEAR review, validation, and batches."""

from __future__ import annotations

import argparse
import concurrent.futures
import faulthandler
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .config import load_config
from .contracts import PaperMetadata, ReviewRequest
from .diffusion_forecast import (
    validate_runtime_replay,
    validate_structural_head_replay,
)
from .env import subprocess_environment
from .graph_action_policy import GRAPH_ACTIONS, RandomizedGraphActionSelector
from .paper_extraction import configured_paper_extractor
from .review_contracts import ReviewBundle, VerificationIssue
from .review_pipeline import ServiceRegistry, review_paper
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


@contextmanager
def _hang_diagnostic(output_dir: Path | None) -> Iterator[None]:
    """Persist Python stacks for unexpectedly long review workers."""
    if output_dir is None:
        yield
        return
    path = output_dir.resolve() / "hang_diagnostic.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = False
    with path.open("w", encoding="utf-8") as handle:
        faulthandler.dump_traceback_later(900, repeat=True, file=handle)
        try:
            yield
            completed = True
        finally:
            faulthandler.cancel_dump_traceback_later()
    if path.stat().st_size == 0:
        path.unlink(missing_ok=True)
    elif completed:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                "\nGEAR_DIAGNOSTIC_RESOLUTION: review command completed normally.\n"
            )


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
    services = None
    if args.forced_graph_action is not None:
        if args.output_dir is None:
            raise ValueError("forced Graph actions require --output-dir")
        services = ServiceRegistry(
            evidence_store=EvidenceStore(args.output_dir.resolve()),
            graph_action_selector=RandomizedGraphActionSelector(
                args.forced_graph_action, args.action_propensity
            ),
            paper_extractor=configured_paper_extractor(config),
        )
    with _hang_diagnostic(args.output_dir):
        bundle = review_paper(
            request,
            output_dir=args.output_dir,
            config=config,
            services=services,
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
    report_dir = (args.batch_report_dir or args.output_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
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
    (report_dir / "batch_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {"case_count": len(results), "status_counts": counts, "results": results}
    (report_dir / "batch_summary.json").write_text(
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
    with _case_lock(args.output_dir, case_id):
        return _run_locked_benchmark_case(started, case, args, output, base)


def _run_locked_benchmark_case(
    started: float,
    case: dict[str, Any],
    args: argparse.Namespace,
    output: Path,
    base: dict[str, Any],
) -> dict[str, Any]:
    """Run one case while its cross-process lock is held."""
    existing = _existing_case_result(output)
    if (
        args.resume
        and existing is not None
        and not (args.retry_failed and existing["status"] == "failed")
    ):
        return {**base, **existing, "elapsed_seconds": 0.0, "resumed": True}
    archived_attempt = _archive_prior_attempt(output)
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
        if case.get("assigned_action"):
            command.extend(
                [
                    "--forced-graph-action",
                    str(case["assigned_action"]),
                    "--action-propensity",
                    str(case["propensity"]),
                ]
            )
        if not args.full_artifacts:
            command.append("--compact-artifacts")
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=subprocess_environment(),
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
            "archived_attempt": str(archived_attempt) if archived_attempt else None,
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
            "archived_attempt": str(archived_attempt) if archived_attempt else None,
            "elapsed_seconds": time.monotonic() - started,
            "resumed": False,
        }


def _archive_prior_attempt(output: Path) -> Path | None:
    """Preserve a prior attempt before retrying into a fresh EvidenceStore."""
    if not output.is_dir() or not any(output.iterdir()):
        return None
    archive_root = output.parent / ".attempt_archive" / output.name
    archive_root.mkdir(parents=True, exist_ok=True)
    archived = archive_root / f"attempt_{time.time_ns()}"
    output.replace(archived)
    return archived


@contextmanager
def _case_lock(output_dir: Path, case_id: str) -> Iterator[None]:
    """Serialize writers for a case across benchmark processes on this host."""
    lock_dir = output_dir.resolve() / ".case_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(case_id.encode("utf-8")).hexdigest() + ".lock"
    with (lock_dir / lock_name).open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
    structural_manifest = config.resolved_structural_head_manifest()
    if structural_manifest is None:
        report["structural_head_release"] = "not_configured"
    else:
        report["structural_head_replay"] = validate_structural_head_replay(
            structural_manifest,
            config.resolved_forecast_release_manifest(),
            config.resolved_forecast_runtime_manifest(),
        )
    report["claim_attribution_mode"] = config.claim_attribution.mode
    attribution_manifest = config.resolved_claim_attribution_manifest()
    if attribution_manifest is None:
        report["claim_attribution_release"] = "deterministic_t0_baseline_no_asset"
    else:
        from .claim_attribution import validate_claim_attribution_replay

        report["claim_attribution_replay"] = validate_claim_attribution_replay(
            attribution_manifest
        )
    action_policy_manifest = config.resolved_graph_action_policy_manifest()
    report["graph_action_policy_enabled"] = config.graph_guidance.action_policy_enabled
    if action_policy_manifest is None:
        report["graph_action_policy_release"] = "not_configured"
    else:
        from .graph_action_policy import validate_graph_action_policy_replay

        report["graph_action_policy_replay"] = validate_graph_action_policy_replay(
            action_policy_manifest
        )
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
    review.add_argument("--forced-graph-action", choices=("baseline", *GRAPH_ACTIONS))
    review.add_argument("--action-propensity", type=float, default=1.0)
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
    benchmark.add_argument(
        "--batch-report-dir",
        type=Path,
        help="Write this invocation's summary outside the shared case output directory.",
    )
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

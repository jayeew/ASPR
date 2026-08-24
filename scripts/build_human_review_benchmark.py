#!/usr/bin/env python3
"""Create frozen, AI-comparable human StructuredReview labels from review histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gear.evaluation.contracts import (
    RevisionIssueDraft,
    RevisionIssueLabel,
)
from gear.codex_cli import CodexCliJsonClient
from gear.config import OpenAICompatibleEndpoint, load_config
from gear.contracts import (
    PaperMetadata,
    ReviewRequest,
    StrictModel,
)
from gear.env import getenv
from gear.model_client import (
    JsonModelClient,
    ModelClientUnavailableError,
)
from gear.openai_compatible_api import OpenAICompatibleJsonClient
from gear.paper_compiler import PaperCompiler
from gear.review_contracts import StructuredReview
from gear.trace import sha256_file

DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data/nature_markdown"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/gear/human_review_reconstruction"
PAPER_IDS_ENV = "ASPR_GEAR_RECONSTRUCTION_PAPER_IDS"
DEFAULT_API_MODELS = (
    "sensenova-6.8-flash-lite",
    "deepseek-v4-flash",
    "glm-5.2",
)

PROMPT = """You extract one final, consolidated human peer review from its complete
transparent review history. The input contains every reviewer round and every author
reply. Synthesize the final residual evaluation: a concern that an author response
adequately addressed and a later reviewer accepted must not appear as a weakness or
question. Retain only unresolved, partially resolved, or enduring limitations in the
final StructuredReview. Use author replies to decide resolution, but do not attribute
opinions to authors. Merge duplicate reviewer concerns. Produce a concise, useful
scientific review—not a transcript. Do not include acceptance/rejection or editorial
decision language. Use only the supplied P:S-* paper evidence keys. Every major or
critical point needs evidence. In a separate revision_issues array, retain concise
labels for persists, partially_resolved, resolved, and unverifiable concerns. A
resolved concern must not re-enter the final review. Reviewer quotations and unique
reviewer bindings are not required. Return only the requested JSON object.
"""


class HumanBenchmarkExtraction(StrictModel):
    review: StructuredReview
    revision_issues: list[RevisionIssueDraft]


def _paper_ids(cli_ids: list[str] | None) -> list[str]:
    """Resolve explicit paper IDs from CLI or a comma-separated environment list."""
    values = cli_ids or [
        value.strip() for value in os.getenv(PAPER_IDS_ENV, "").split(",")
    ]
    values = [value for value in values if value]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"provide unique IDs via --paper-id or {PAPER_IDS_ENV}")
    return values


def _strict_schema(paper_id: str, evidence_keys: list[str]) -> dict[str, Any]:
    """Constrain the review and revision labels to immutable paper spans."""
    schema = HumanBenchmarkExtraction.model_json_schema()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("default", None)
            if isinstance(value.get("properties"), dict):
                value["required"] = list(value["properties"])
                value["additionalProperties"] = False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    definitions = schema["$defs"]
    definitions["StructuredReview"]["properties"]["paper_id"] = {
        "enum": [paper_id],
        "type": "string",
    }
    definitions["ReviewSummary"]["properties"]["evidence_keys"]["items"] = {
        "enum": evidence_keys,
        "type": "string",
    }
    definitions["ReviewPoint"]["properties"]["evidence_keys"]["items"] = {
        "enum": evidence_keys,
        "type": "string",
    }
    definitions["RevisionIssueDraft"]["properties"]["paper_evidence_keys"]["items"] = {
        "enum": evidence_keys,
        "type": "string",
    }
    return schema


def _history_prompt(paper_ir: Any, review_path: Path) -> str:
    """Build the only model input: final manuscript evidence and complete history."""
    evidence = [
        {
            "evidence_key": f"P:{span.span_id}",
            "section_path": list(span.section_path),
            "text": span.text,
        }
        for span in paper_ir.spans
    ]
    return json.dumps(
        {
            "paper_id": paper_ir.paper_id,
            "paper_evidence": evidence,
            "complete_transparent_review_history": review_path.read_text(
                encoding="utf-8", errors="replace"
            ),
        },
        ensure_ascii=False,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: Any) -> None:
    """Atomically persist one resumable case result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalize_novelty_judgment(payload: dict[str, Any]) -> None:
    """Apply deterministic novelty invariants without changing review meaning."""
    novelty = payload["novelty"]
    for bucket in ("supporting_points", "limiting_points", "uncertain_points"):
        for point in novelty.get(bucket, []):
            if point.get("aspect") not in {"novelty_prior_art", "contribution"}:
                point["aspect"] = "novelty_prior_art"
    supporting = bool(novelty["supporting_points"])
    limiting = bool(novelty["limiting_points"])
    uncertain = bool(novelty["uncertain_points"])
    novelty["judgment"] = (
        "mixed"
        if sum((supporting, limiting, uncertain)) > 1
        else (
            "positive"
            if supporting
            else (
                "negative"
                if limiting
                else "uncertain" if uncertain else "not_discussed"
            )
        )
    )


def _issue_id(paper_id: str, issue: RevisionIssueDraft) -> str:
    normalized = re.sub(r"\W+", " ", issue.text.casefold()).strip()
    identity = f"{paper_id}|{issue.status}|{normalized}"
    return "RI-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:18]


def _paper_metadata(raw_id: str, metadata_root: Path | None) -> PaperMetadata:
    if metadata_root is None:
        return PaperMetadata(doi=f"10.1038/{raw_id}")
    path = metadata_root / f"{raw_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing paper metadata for {raw_id}: {path}")
    return PaperMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def _source_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("source manifest must contain a JSON list")
    return [dict(row) for row in payload]


def _source_metadata(raw_id: str, rows: list[dict[str, Any]]) -> PaperMetadata | None:
    for row in rows:
        if row.get("source_paper_id") == raw_id:
            return PaperMetadata(
                doi=row.get("doi") or f"10.1038/{raw_id}",
                openalex_id=row.get("openalex_id"),
            )
    return None


def _model_clients(
    api_models: list[str],
    api_base_url: str,
    api_key_env: str,
    codex_fallback: bool,
) -> list[tuple[str, JsonModelClient]]:
    """Build lazy stateless clients in the requested failover order."""
    clients: list[tuple[str, JsonModelClient]] = []
    if api_base_url and api_key_env:
        for model in api_models:
            endpoint = OpenAICompatibleEndpoint(
                base_url=api_base_url,
                model=model,
                api_key_env=api_key_env,
                timeout_seconds=int(getenv("ASPR_GEAR_API_TIMEOUT_SECONDS", "1800")),
            )
            clients.append((model, OpenAICompatibleJsonClient(endpoint)))
    if codex_fallback:
        config = load_config()
        clients.append(
            (
                f"codex_cli:{config.codex_cli.model}",
                CodexCliJsonClient(
                    config.codex_cli,
                    cache_dir=config.resolve_path(config.cache_dir) / "model_responses",
                ),
            )
        )
    if not clients:
        raise ValueError("no model client configured")
    return clients


@contextmanager
def _wall_clock_limit(seconds: int) -> Any:
    """Bound one blocking provider request on Unix batch workers."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"model request exceeded {seconds} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _extract_with_failover(
    clients: list[tuple[str, JsonModelClient]],
    *,
    disabled: set[str],
    paper_id: str,
    evidence_keys: list[str],
    user: str,
    attempt_timeout_seconds: int,
) -> tuple[HumanBenchmarkExtraction, str, list[dict[str, str]]]:
    attempts: list[dict[str, str]] = []
    for model, client in clients:
        if model in disabled:
            continue
        try:
            timeout_seconds = (
                max(900, attempt_timeout_seconds)
                if model.startswith("codex_cli:")
                else attempt_timeout_seconds
            )
            with _wall_clock_limit(timeout_seconds):
                payload = client.generate_json(
                    system=PROMPT,
                    user=user,
                    response_schema=_strict_schema(paper_id, evidence_keys),
                )
            if isinstance(payload.get("review"), dict):
                _normalize_novelty_judgment(payload["review"])
            extraction = HumanBenchmarkExtraction.model_validate(payload)
            review = extraction.review
            known_keys = set(evidence_keys)
            used = {key for point in review.all_points() for key in point.evidence_keys}
            issue_keys = {
                key
                for issue in extraction.revision_issues
                for key in issue.paper_evidence_keys
            }
            if review.paper_id != paper_id:
                raise ValueError("model returned a mismatched paper ID")
            if not set(review.summary.evidence_keys).issubset(known_keys):
                raise ValueError("summary returned an unknown paper evidence key")
            if not used.issubset(known_keys) or not issue_keys.issubset(known_keys):
                raise ValueError("model returned an unknown paper evidence key")
            return extraction, model, attempts
        except (ModelClientUnavailableError, TimeoutError) as exc:
            disabled.add(model)
            attempts.append({"model": model, "error": f"{type(exc).__name__}:{exc}"})
        except (KeyError, TypeError, ValueError) as exc:
            attempts.append({"model": model, "error": f"{type(exc).__name__}:{exc}"})
    raise ModelClientUnavailableError(json.dumps(attempts, ensure_ascii=False))


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    HumanBenchmarkExtraction.model_validate(payload["extraction"])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", action="append")
    parser.add_argument("--dataset-id", default="human_review_benchmark")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--api-model", action="append")
    parser.add_argument("--api-base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--attempt-timeout-seconds", type=int, default=300)
    parser.add_argument("--no-codex-fallback", action="store_true")
    parser.add_argument("--checkpoint-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = (args.output_root / args.dataset_id).resolve()
    if output_dir.exists():
        if args.resume and (output_dir / "release_manifest.json").is_file():
            return 0
        raise FileExistsError(f"refusing to overwrite frozen release: {output_dir}")
    work_dir = args.output_root.resolve() / f".{args.dataset_id}.work"
    if work_dir.exists() and not args.resume:
        raise FileExistsError(
            f"partial reconstruction exists; use --resume: {work_dir}"
        )
    source_rows = _source_rows(args.source_manifest)
    raw_ids = (
        _paper_ids(args.paper_id)
        if args.paper_id
        else [str(row["source_paper_id"]) for row in source_rows]
    )
    if not raw_ids:
        raise ValueError("no paper IDs resolved")
    api_models = args.api_model or list(DEFAULT_API_MODELS)
    api_base_url = (
        "" if args.skip_api else (args.api_base_url or getenv("ASPR_GEAR_API_BASE_URL"))
    )
    api_key_env = args.api_key_env or getenv(
        "ASPR_GEAR_API_KEY_ENV", "SENSENOVA_API_KEY"
    )
    clients = _model_clients(
        api_models,
        api_base_url,
        api_key_env,
        not args.no_codex_fallback,
    )
    disabled: set[str] = set()
    compiler = PaperCompiler(load_config())
    reviews: list[dict[str, Any]] = []
    revision_labels: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    model_counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    for raw_id in raw_ids:
        checkpoint_path = work_dir / "cases" / f"{raw_id}.json"
        try:
            checkpoint = _load_checkpoint(checkpoint_path) if args.resume else None
        except (KeyError, OSError, TypeError, ValueError):
            checkpoint = None
        if checkpoint is not None:
            extraction = HumanBenchmarkExtraction.model_validate(
                checkpoint["extraction"]
            )
            model_used = str(checkpoint["model_id"])
            source = dict(checkpoint["source"])
            model_counts[model_used] += 1
            reviews.append(extraction.review.model_dump(mode="json"))
            revision_labels.extend(checkpoint["revision_labels"])
            sources.append(source)
            continue
        paper_path = args.source_root / "paper" / f"{raw_id}.md"
        review_path = args.source_root / "peer_review" / f"{raw_id}_r.md"
        if not paper_path.is_file() or not review_path.is_file():
            failures.append({"paper_id": raw_id, "error": "missing_markdown_pair"})
            continue
        metadata = _source_metadata(raw_id, source_rows) or _paper_metadata(
            raw_id, args.metadata_root
        )
        paper_ir = compiler.compile(
            ReviewRequest(
                paper_path=paper_path,
                metadata=metadata,
            )
        )
        keys = [f"P:{span.span_id}" for span in paper_ir.spans]
        try:
            extraction, model_used, attempts = _extract_with_failover(
                clients,
                disabled=disabled,
                paper_id=paper_ir.paper_id,
                evidence_keys=keys,
                user=_history_prompt(paper_ir, review_path),
                attempt_timeout_seconds=args.attempt_timeout_seconds,
            )
        except ModelClientUnavailableError as exc:
            failures.append({"paper_id": raw_id, "error": str(exc)})
            continue
        review = extraction.review
        case_revision_labels: list[dict[str, Any]] = []
        for issue in extraction.revision_issues:
            if not set(issue.paper_evidence_keys).issubset(keys):
                raise ValueError(
                    "revision issue returned an unknown paper evidence key"
                )
            case_revision_labels.append(
                RevisionIssueLabel(
                    paper_id=paper_ir.paper_id,
                    issue_id=_issue_id(paper_ir.paper_id, issue),
                    text=issue.text,
                    section=issue.section,
                    aspect=issue.aspect,
                    severity=issue.severity,
                    status=issue.status,
                    paper_evidence_keys=list(dict.fromkeys(issue.paper_evidence_keys)),
                ).model_dump(mode="json")
            )
        source = {
            "paper_id": paper_ir.paper_id,
            "source_paper_path": str(paper_path.resolve()),
            "source_paper_sha256": sha256_file(paper_path),
            "source_review_history_path": str(review_path.resolve()),
            "source_review_history_sha256": sha256_file(review_path),
        }
        _write_json(
            checkpoint_path,
            {
                "raw_id": raw_id,
                "model_id": model_used,
                "failed_model_attempts": attempts,
                "extraction": extraction.model_dump(mode="json"),
                "revision_labels": case_revision_labels,
                "source": source,
            },
        )
        model_counts[model_used] += 1
        reviews.append(review.model_dump(mode="json"))
        revision_labels.extend(case_revision_labels)
        sources.append(source)
    if not args.checkpoint_only:
        _write_json(
            work_dir / "progress.json",
            {
                "dataset_id": args.dataset_id,
                "requested": len(raw_ids),
                "completed": len(reviews),
                "failures": failures,
                "disabled_models": sorted(disabled),
                "model_counts": dict(model_counts),
            },
        )
    if failures or len(reviews) != len(raw_ids):
        raise RuntimeError(
            f"reconstruction incomplete: {len(reviews)}/{len(raw_ids)}; "
            f"resume from {work_dir}"
        )
    if args.checkpoint_only:
        return 0
    output_dir.mkdir(parents=True)
    _write_jsonl(output_dir / "human_structured_reviews.jsonl", reviews)
    _write_jsonl(output_dir / "revision_issue_labels.jsonl", revision_labels)
    (output_dir / "source_manifest.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "contract": "human_structured_review_benchmark_v2",
        "dataset_id": args.dataset_id,
        "review_contract": "gear.review_contracts.StructuredReview",
        "record_count": len(reviews),
        "revision_issue_label_count": len(revision_labels),
        "model_backend": "openai_compatible_with_codex_cli_fallback",
        "model_ids": list(model_counts),
        "model_record_counts": dict(model_counts),
        "input_mode": f"--paper-id or {PAPER_IDS_ENV}",
        "history_policy": "all reviewer rounds plus author replies; resolved concerns omitted",
        "source_manifest_sha256": sha256_file(output_dir / "source_manifest.json"),
        "human_structured_reviews_sha256": sha256_file(
            output_dir / "human_structured_reviews.jsonl"
        ),
        "revision_issue_labels_sha256": sha256_file(
            output_dir / "revision_issue_labels.jsonl"
        ),
    }
    (output_dir / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

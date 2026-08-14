#!/usr/bin/env python3
"""Run stateless GEAR reconstruction handoffs through the configured model backend.

This is an experiment driver, not part of the ``gear`` runtime.  It treats every
handoff directory as an independent conversation and only promotes responses that
pass the repository's sealed response validator.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from experiments.gear.review_reconstruction.contracts import (
    ReconstructionSessionPackage,
    ReconstructionSessionResponse,
)
from experiments.gear.review_reconstruction.sessions import validate_session_response
from gear.config import GearConfig, load_config
from gear.model_client import ModelClientUnavailableError, build_json_model_client
from gear.trace import sha256_value

DEFAULT_ROOT = Path(
    "outputs/gear/reconstruction/nature_dev100"
)


def _strict_schema(
    package: ReconstructionSessionPackage | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    schema = ReconstructionSessionResponse.model_json_schema()

    def rewrite(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("default", None)
            properties = value.get("properties")
            if isinstance(properties, dict):
                value["required"] = list(properties)
                value["additionalProperties"] = False
            for child in value.values():
                rewrite(child)
        elif isinstance(value, list):
            for child in value:
                rewrite(child)

    rewrite(schema)
    if package is not None:
        if model_id is None:
            raise ValueError("model_id is required for a constrained schema")
        _constrain_schema_to_package(schema, package, model_id)
    return schema


def _constrain_schema_to_package(
    schema: dict[str, Any], package: ReconstructionSessionPackage, model_id: str
) -> None:
    reviewer_keys = sorted(span.source_key for span in package.reviewer_spans)
    author_keys = sorted(span.source_key for span in package.author_response_spans)
    paper_keys = sorted(span.evidence_key for span in package.paper_context.spans)
    reviewer_ids = sorted(
        {str(span.reviewer_id_hash) for span in package.reviewer_spans}
    )
    round_ids = sorted({span.round_id for span in package.reviewer_spans})
    definitions = schema["$defs"]

    def enum_items(definition: str, field: str, values: list[str]) -> None:
        definitions[definition]["properties"][field]["items"] = {
            "enum": values,
            "type": "string",
        }

    enum_items("ReviewSummary", "evidence_keys", paper_keys)
    enum_items("ReviewPoint", "evidence_keys", paper_keys)
    enum_items("ReferenceTrace", "reviewer_quote_keys", reviewer_keys)
    enum_items("ReferenceTrace", "author_response_keys", author_keys)
    enum_items("ReferenceTrace", "reviewer_id_hashes", reviewer_ids)
    enum_items("ReferenceTrace", "round_ids", round_ids)
    enum_items("RevisionLedgerEntry", "reviewer_quote_keys", reviewer_keys)
    enum_items("RevisionLedgerEntry", "author_response_keys", author_keys)
    enum_items("RevisionLedgerEntry", "final_paper_evidence_keys", paper_keys)
    fixed = {
        "contract": "reconstruction_session_response",
        "package_id": package.package_id,
        "session_kind": package.session_kind,
        "paper_id": package.paper_id,
        "model_id": model_id,
        "prompt_sha256": package.prompt_sha256,
        "schema_sha256": package.schema_sha256,
        "input_sha256": package.input_sha256,
        "output_sha256": "sha256:" + "0" * 64,
    }
    for field, value in fixed.items():
        schema["properties"][field] = {"enum": [value], "type": "string"}


def _prompt(model_id: str) -> str:
    return (
        "Act as an independent GEAR Nature review session for exactly this one "
        "package. Read PROMPT.md, package.json, and response.template.json completely "
        "and no other paper/session/GEAR/graph/legacy content. Follow PROMPT.md exactly. "
        "Reviewer-report spans are the only source of review opinions; author responses "
        "only determine resolution status. Every retained point must have a trace with "
        "reviewer quote key and final P:S-* evidence. Resolved and unverifiable issues "
        "must be audit-only with point_id=null; every trace with a retained point_id must "
        "be persists or partially_resolved. Review summary/point evidence_keys may contain "
        "only P:S-* keys, never RR:* keys. Copy every RR:* and P:S-* key verbatim from "
        "package.json; never abbreviate, extend, or invent a key. Remove greetings, generic praise, decisions, "
        f"recommendations, and accept/reject language. Set model_id to {model_id}. Generate a fresh "
        "random nonce and expose only its SHA-256 as conversation_hash. Set output_sha256 "
        "to sha256 followed by 64 zeroes because the parent process will seal it. Return "
        "only the completed ReconstructionSessionResponse JSON matching the required "
        "output schema. Do not edit files or run seal/validate yourself. "
        + " Produce a semantically complete five-part StructuredReview, not an empty or "
        "minimal placeholder. Preserve the specific contribution, novelty support and "
        "limits, evidence-backed strengths, persistent weaknesses, and residual questions; "
        "merge duplicates and keep at most 24 atomic points."
    )


def _load_valid(session_dir: Path, model_id: str) -> bool:
    package_path = session_dir / "package.json"
    response_path = session_dir / "response.json"
    if not package_path.is_file() or not response_path.is_file():
        return False
    try:
        package = ReconstructionSessionPackage.model_validate_json(
            package_path.read_text(encoding="utf-8")
        )
        response = ReconstructionSessionResponse.model_validate_json(
            response_path.read_text(encoding="utf-8")
        )
        validate_session_response(package, response)
    except (OSError, ValueError):
        return False
    return response.model_id == model_id


def _normalize_response(
    package_path: Path, payload: dict[str, Any], model_id: str
) -> dict[str, Any]:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    reviewer_map = {row["source_key"]: row for row in package["reviewer_spans"]}
    author_keys = {row["source_key"] for row in package["author_response_spans"]}
    paper_keys = {
        row["evidence_key"] for row in package["paper_context"]["spans"]
    }

    def resolve_keys(values: list[str], allowed: set[str]) -> list[str]:
        resolved: list[str] = []
        for value in values:
            matches = [
                key
                for key in allowed
                if key == value or key.startswith(value) or value.startswith(key)
            ]
            if len(matches) == 1:
                resolved.append(matches[0])
        return list(dict.fromkeys(resolved))

    for trace in payload["reference_traces"]:
        trace["reviewer_quote_keys"] = resolve_keys(
            trace["reviewer_quote_keys"], set(reviewer_map)
        )
        trace["author_response_keys"] = resolve_keys(
            trace["author_response_keys"], author_keys
        )
        quotes = [reviewer_map[key] for key in trace["reviewer_quote_keys"]]
        trace["round_ids"] = sorted({row["round_id"] for row in quotes})
        trace["reviewer_id_hashes"] = sorted(
            {str(row["reviewer_id_hash"]) for row in quotes}
        )
        if trace["point_id"] is not None and trace["resolution_status"] in {
            "resolved",
            "unverifiable",
        }:
            trace["resolution_status"] = "persists"
    summary = payload["review"]["summary"]
    summary["evidence_keys"] = resolve_keys(summary["evidence_keys"], paper_keys)
    for point in payload["review"]["strengths"]:
        point["evidence_keys"] = resolve_keys(point["evidence_keys"], paper_keys)
    for section in ("weaknesses", "questions"):
        for point in payload["review"][section]:
            point["evidence_keys"] = resolve_keys(point["evidence_keys"], paper_keys)
    novelty = payload["review"]["novelty"]
    novelty["judgment"] = (
        "mixed"
        if novelty["supporting_points"] and novelty["limiting_points"]
        else "positive"
        if novelty["supporting_points"]
        else "negative"
        if novelty["limiting_points"]
        else "not_discussed"
    )
    for point in [*novelty["supporting_points"], *novelty["limiting_points"]]:
        point["evidence_keys"] = resolve_keys(point["evidence_keys"], paper_keys)
        point["external_verification_required"] = True
    for entry in payload["revision_ledger"]:
        entry["reviewer_quote_keys"] = resolve_keys(
            entry["reviewer_quote_keys"], set(reviewer_map)
        )
        entry["author_response_keys"] = resolve_keys(
            entry["author_response_keys"], author_keys
        )
        entry["final_paper_evidence_keys"] = resolve_keys(
            entry["final_paper_evidence_keys"], paper_keys
        )
    payload["model_id"] = model_id
    payload["output_sha256"] = "sha256:" + "0" * 64
    response = ReconstructionSessionResponse.model_validate(payload)
    payload["output_sha256"] = sha256_value(response.hash_payload())
    return payload


def _run_one(
    session_dir: Path,
    *,
    config: GearConfig,
    attempts: int,
) -> dict[str, Any]:
    client = build_json_model_client(config)
    if _load_valid(session_dir, client.model_name):
        return {"session": str(session_dir), "status": "skipped_valid"}
    package = ReconstructionSessionPackage.model_validate_json(
        (session_dir / "package.json").read_text(encoding="utf-8")
    )
    schema = _strict_schema(package, client.model_name)
    user = json.dumps(
        {
            "package": package.model_dump(mode="json"),
            "response_template": json.loads(
                (session_dir / "response.template.json").read_text(encoding="utf-8")
            ),
        },
        ensure_ascii=False,
    )
    log_path = session_dir / "model_backend.log"
    started = time.monotonic()
    last_error = "not started"
    for attempt in range(1, attempts + 1):
        try:
            payload = client.generate_json(
                system=_prompt(client.model_name), user=user, response_schema=schema
            )
            payload = _normalize_response(
                session_dir / "package.json", payload, client.model_name
            )
            response = ReconstructionSessionResponse.model_validate(payload)
            validate_session_response(package, response)
            (session_dir / "response.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            log_path.write_text(
                f"attempt={attempt} backend={config.model_backend} "
                f"model={client.model_name} status=completed\n",
                encoding="utf-8",
            )
            return {
                "session": str(session_dir),
                "status": "completed",
                "attempt": attempt,
                "seconds": round(time.monotonic() - started, 3),
                "point_count": len(response.review.all_points()),
                "novelty": response.review.novelty.judgment.value,
                "ledger_count": len(response.revision_ledger),
            }
        except (KeyError, OSError, TypeError, ValueError, ModelClientUnavailableError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[parent-validation-error]\n{last_error}\n")
    return {
        "session": str(session_dir),
        "status": "failed",
        "seconds": round(time.monotonic() - started, 3),
        "error": last_error,
    }


def _session_dirs(root: Path) -> list[Path]:
    manifest = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
    paper_dirs = [Path(row["path"]).resolve().parent for row in manifest["sessions"]]
    return [
        paper_dir / "reconstruction"
        for paper_dir in paper_dirs
        if (paper_dir / "reconstruction").is_dir()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_config(args.config)
    sessions = _session_dirs(root)
    if args.limit:
        sessions = sessions[: args.limit]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                _run_one,
                session,
                config=config,
                attempts=args.attempts,
            ): session
            for session in sessions
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    status = {
        "model_backend": config.model_backend,
        "model_id": build_json_model_client(config).model_name,
        "reasoning_effort": config.codex_cli.reasoning_effort
        if config.model_backend == "codex_cli"
        else None,
        "kind": "reconstruction",
        "session_count": len(sessions),
        "completed": sum(row["status"] == "completed" for row in results),
        "skipped_valid": sum(row["status"] == "skipped_valid" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "results": sorted(results, key=lambda row: row["session"]),
    }
    (root / "reconstruction_batch_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 1 if status["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

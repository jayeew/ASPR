#!/usr/bin/env python3
"""Prepare and run blinded point-match judges for the Nature dev100 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from experiments.gear.review_reconstruction.contracts import (
    ReconstructionSessionResponse,
)
from experiments.gear.review_reconstruction.evaluation import (
    MatchJudgePackage,
    MatchJudgeResponse,
    build_blind_match_package,
    validate_match_judge_response,
)
from gear.review_contracts import StructuredReview
from gear.config import GearConfig, load_config
from gear.model_client import ModelClientUnavailableError, build_json_model_client

DEFAULT_RECONSTRUCTION_ROOT = Path(
    "outputs/gear/reconstruction/nature_dev100"
)
DEFAULT_AGENT_ROOT = Path(
    "outputs/gear/consistency/nature_dev100/agent_reviews"
)
DEFAULT_JUDGE_ROOT = Path(
    "outputs/gear/consistency/nature_dev100/match_judging"
)
def _paper_dirs(root: Path) -> list[Path]:
    manifest = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
    return [Path(row["path"]).resolve().parent for row in manifest["sessions"]]


def _reference_review(paper_dir: Path) -> StructuredReview:
    response_path = paper_dir / "reconstruction" / "response.json"
    if not response_path.is_file():
        raise FileNotFoundError(f"reconstruction response missing: {response_path}")
    response = ReconstructionSessionResponse.model_validate_json(
        response_path.read_text(encoding="utf-8")
    )
    return response.review


def _agent_review(agent_root: Path, slug: str, paper_id: str) -> StructuredReview:
    path = agent_root / slug / "review.json"
    review = StructuredReview.model_validate_json(path.read_text(encoding="utf-8"))
    return review.model_copy(update={"paper_id": paper_id})


def _strict_schema(package: MatchJudgePackage, model_id: str) -> dict[str, Any]:
    schema = MatchJudgeResponse.model_json_schema()

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
    schema["properties"]["task_id"] = {"enum": [package.task_id], "type": "string"}
    schema["properties"]["model_id"] = {"enum": [model_id], "type": "string"}
    decision = schema["$defs"]["PointMatchDecision"]["properties"]
    decision["paper_id"] = {"enum": [package.paper_id_hash], "type": "string"}
    decision["reference_point_id"] = {
        "enum": sorted({left for left, _ in package.candidate_pairs}),
        "type": "string",
    }
    decision["candidate_point_id"] = {
        "enum": sorted({right for _, right in package.candidate_pairs}),
        "type": "string",
    }
    return schema


def _write_empty_response(package: MatchJudgePackage, target: Path) -> None:
    nonce_hash = hashlib.sha256(secrets.token_bytes(32)).hexdigest()
    response = MatchJudgeResponse(
        task_id=package.task_id,
        model_id="deterministic-no-pairs",
        conversation_hash=f"sha256:{nonce_hash}",
        decisions=[],
    )
    validate_match_judge_response(package, response)
    target.write_text(response.model_dump_json(indent=2) + "\n", encoding="utf-8")


def prepare(
    reconstruction_root: Path,
    agent_root: Path,
    judge_root: Path,
    config: GearConfig,
) -> dict[str, Any]:
    built = 0
    empty = 0
    errors: list[str] = []
    for paper_dir in _paper_dirs(reconstruction_root):
        slug = paper_dir.name
        try:
            reference = _reference_review(paper_dir)
            candidate = _agent_review(agent_root, slug, reference.paper_id)
            package = build_blind_match_package(reference, candidate, top_k=5)
            target = judge_root / slug
            target.mkdir(parents=True, exist_ok=True)
            (target / "package.json").write_text(
                package.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            (target / "response.schema.json").write_text(
                json.dumps(
                    _strict_schema(
                        package, build_json_model_client(config).model_name
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if not package.candidate_pairs:
                _write_empty_response(package, target / "response.json")
                empty += 1
            built += 1
        except (FileNotFoundError, OSError, ValueError) as exc:
            errors.append(f"{slug}:{type(exc).__name__}:{exc}")
    return {"built": built, "empty_pair_packages": empty, "errors": errors}


def _load_valid(session_dir: Path) -> bool:
    try:
        package = MatchJudgePackage.model_validate_json(
            (session_dir / "package.json").read_text(encoding="utf-8")
        )
        response = MatchJudgeResponse.model_validate_json(
            (session_dir / "response.json").read_text(encoding="utf-8")
        )
        validate_match_judge_response(package, response)
    except (FileNotFoundError, OSError, ValueError):
        return False
    return True


def _prompt() -> str:
    return (
        "Blindly evaluate every listed candidate pair in the supplied package. "
        "SAME_POINT requires the same "
        "atomic scientific proposition and direction; PARTIAL_POINT shares a material "
        "subset but differs in scope; CONTRADICTORY has opposite direction; otherwise "
        "NO_MATCH. Cover every candidate_pairs tuple exactly once and add no other pair. "
        "Use paper_id_hash as paper_id. Set model_id to the configured model ID. "
        "Generate a fresh random nonce and expose only its SHA-256 as conversation_hash. "
        "Return JSON only."
    )


def _run_one(session_dir: Path, config: GearConfig, attempts: int) -> dict[str, Any]:
    if _load_valid(session_dir):
        return {"slug": session_dir.name, "status": "skipped_valid"}
    package = MatchJudgePackage.model_validate_json(
        (session_dir / "package.json").read_text(encoding="utf-8")
    )
    client = build_json_model_client(config)
    schema = _strict_schema(package, client.model_name)
    user = package.model_dump_json()
    last_error = "not started"
    started = time.monotonic()
    for attempt in range(1, attempts + 1):
        try:
            payload = client.generate_json(
                system=_prompt(), user=user, response_schema=schema
            )
            payload["model_id"] = client.model_name
            response = MatchJudgeResponse.model_validate(payload)
            validate_match_judge_response(package, response)
            (session_dir / "response.json").write_text(
                response.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            (session_dir / "model_backend.log").write_text(
                f"attempt={attempt} backend={config.model_backend} "
                f"model={client.model_name} status=completed\n",
                encoding="utf-8",
            )
            return {
                "slug": session_dir.name,
                "status": "completed",
                "attempt": attempt,
                "pairs": len(response.decisions),
                "seconds": round(time.monotonic() - started, 3),
            }
        except (ModelClientUnavailableError, OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return {"slug": session_dir.name, "status": "failed", "error": last_error}


def run(judge_root: Path, config: GearConfig, jobs: int, attempts: int) -> dict[str, Any]:
    sessions = sorted(path.parent for path in judge_root.glob("*/package.json"))
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_run_one, path, config, attempts): path
            for path in sessions
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    return {
        "model_backend": config.model_backend,
        "model_id": build_json_model_client(config).model_name,
        "reasoning_effort": config.codex_cli.reasoning_effort
        if config.model_backend == "codex_cli"
        else None,
        "session_count": len(sessions),
        "completed": sum(row["status"] == "completed" for row in results),
        "skipped_valid": sum(row["status"] == "skipped_valid" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "results": sorted(results, key=lambda row: row["slug"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--reconstruction-root", type=Path, default=DEFAULT_RECONSTRUCTION_ROOT)
    parser.add_argument("--agent-root", type=Path, default=DEFAULT_AGENT_ROOT)
    parser.add_argument("--judge-root", type=Path, default=DEFAULT_JUDGE_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    judge_root = args.judge_root.resolve()
    judge_root.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        result = prepare(
            args.reconstruction_root.resolve(),
            args.agent_root.resolve(),
            judge_root,
            config,
        )
    else:
        result = run(
            judge_root, config, args.jobs, args.attempts
        )
    (judge_root / f"{args.command}_status.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("errors") or result.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())

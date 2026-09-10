#!/usr/bin/env python3
"""Extend to 1,000 papers and resume claims/reviews with gpt-5.6-luna / low / fast."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = ROOT / "experiments" / "innovation_200"
sys.path.insert(0, str(ROOT))

from experiments.innovation_200.common import read_jsonl, write_json
from experiments.innovation_200.contracts import HumanReferenceSet, PaperRow
from gear.env import subprocess_environment
from gear.innovation.contracts import ClaimSet

DEFAULT_STUDY = ROOT / "outputs" / "innovation_200_20260907"
DEFAULT_MANIFEST = ROOT / "data" / "nature_2026_testset" / "manifest.jsonl"
RUN_MODEL = "gpt-5.6-luna"
RUN_REASONING_EFFORT = "low"
RUN_SERVICE_TIER = "fast"


def logger_for(study: Path) -> logging.Logger:
    logger = logging.getLogger("innovation_200.run_claims_reviews_1000")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    log_path = study / "logs" / "run_claims_reviews_1000.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8", mode="a"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def acquire_lock(study: Path) -> TextIO:
    lock_path = study / "status" / "run_claims_reviews_1000.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError("another 1,000-paper claims/reviews run is active") from exc
    return handle


def paper_row(source: dict[str, object], metadata: dict[str, object]) -> PaperRow:
    return PaperRow.model_validate(
        {
            "paper_id": str(source["article_id"]),
            "title": str(source["title"]),
            "doi": str(source["doi"]),
            "journal_id": str(source["journal_id"]),
            "journal_name": str(source["journal_name"]),
            "publication_date": str(source["publication_date"])[:10],
            "paper_path": str(Path(str(source["paper_markdown_path"])).resolve()),
            "review_path": str(
                Path(str(source["peer_review_markdown_path"])).resolve()
            ),
            "field_name": str(metadata["field_name"]),
            "field_source": metadata["field_source"],
            "abstract_text": str(metadata["abstract_text"]),
            "openalex_work_id": metadata.get("openalex_work_id"),
            "reference_work_ids": metadata.get("reference_work_ids", []),
            "authors": metadata.get("authors", []),
        }
    )


def atomic_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def extend_roster(study: Path, manifest_path: Path) -> list[dict[str, object]]:
    source = read_jsonl(manifest_path)
    if len(source) != 1_000:
        raise ValueError(f"expected 1,000 manifest rows, found {len(source)}")
    source_by_id = {str(row["article_id"]): row for row in source}
    if len(source_by_id) != 1_000:
        raise ValueError("manifest article IDs are not unique")
    existing = read_jsonl(study / "papers.jsonl")
    existing_ids = [str(row["paper_id"]) for row in existing]
    if len(existing_ids) not in {200, 1_000} or len(set(existing_ids)) != len(
        existing_ids
    ):
        raise ValueError("active papers.jsonl must contain 200 or 1,000 unique rows")
    unknown = set(existing_ids) - set(source_by_id)
    if unknown:
        raise ValueError(
            f"active roster contains unknown paper IDs: {sorted(unknown)[:3]}"
        )
    rows_by_id = {str(row["paper_id"]): row for row in existing}
    for paper_id, row in source_by_id.items():
        metadata_path = study / "metadata" / f"{paper_id}.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing cached metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows_by_id[paper_id] = paper_row(row, metadata).model_dump(mode="json")
    order = existing_ids + sorted(set(source_by_id) - set(existing_ids))
    roster = [rows_by_id[paper_id] for paper_id in order]
    atomic_jsonl(study / "papers.jsonl", roster)
    write_distribution(study, roster)
    return roster


def write_distribution(study: Path, rows: list[dict[str, object]]) -> None:
    write_json(
        study / "full_corpus_distribution.json",
        {
            "paper_count": len(rows),
            "journal_actual": dict(Counter(str(row["journal_name"]) for row in rows)),
            "field_actual": dict(Counter(str(row["field_name"]) for row in rows)),
            "field_source": dict(Counter(str(row["field_source"]) for row in rows)),
        },
    )


def valid_claim(study: Path, paper_id: str) -> bool:
    path = study / "papers" / paper_id / "shared" / "claims.json"
    try:
        saved = ClaimSet.model_validate_json(path.read_text(encoding="utf-8"))
        return saved.paper_id == paper_id
    except (OSError, ValueError, TypeError):
        return False


def valid_reference(study: Path, paper_id: str) -> bool:
    path = study / "human_refs" / f"{paper_id}.json"
    try:
        saved = HumanReferenceSet.model_validate_json(path.read_text(encoding="utf-8"))
        return saved.paper_id == paper_id
    except (OSError, ValueError, TypeError):
        return False


def completion(study: Path, paper_ids: list[str]) -> tuple[int, int]:
    claims = sum(valid_claim(study, paper_id) for paper_id in paper_ids)
    references = sum(valid_reference(study, paper_id) for paper_id in paper_ids)
    return claims, references


def stage_command(name: str, study: Path, workers: int, cli_limit: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(SCRIPT_ROOT / name),
        "--study",
        str(study),
        "--workers",
        str(workers),
        "--cli-limit",
        str(cli_limit),
    ]


def stage_environment() -> dict[str, str]:
    """Pin both endpoint and role routing for these two child stages only."""
    env = subprocess_environment()
    env.update(
        PYTHONUNBUFFERED="1",
        ASPR_GEAR_MODEL_BACKEND="codex_cli",
        ASPR_GEAR_CODEX_MODEL=RUN_MODEL,
        ASPR_GEAR_CODEX_REASONING_EFFORT=RUN_REASONING_EFFORT,
        GEAR_STUDY_MODEL=RUN_MODEL,
        GEAR_STUDY_REASONING_EFFORT=RUN_REASONING_EFFORT,
        GEAR_CODEX_SERVICE_TIER=RUN_SERVICE_TIER,
        GEAR_MODEL_RETRIES="2",
    )
    return env


def run_round(study: Path, workers: int, cli_limit: int, env: dict[str, str]) -> None:
    commands = {
        "claims": stage_command("extract_claims.py", study, workers, cli_limit),
        "reviews": stage_command("reconstruct_reviews.py", study, workers, cli_limit),
    }
    processes = {
        name: subprocess.Popen(command, cwd=ROOT, env=env)
        for name, command in commands.items()
    }
    try:
        return_codes = {name: process.wait() for name, process in processes.items()}
    except KeyboardInterrupt:
        terminate_processes(processes.values())
        raise
    failures = {name: code for name, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"stage process failures: {failures}")


def terminate_processes(processes: Iterable[subprocess.Popen[bytes]]) -> None:
    children = list(processes)
    for process in children:
        if process.poll() is None:
            process.terminate()
    for process in children:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--cli-limit", type=int, default=48)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 128 or not 1 <= args.cli_limit <= 64:
        raise ValueError("workers must be 1..128 and cli-limit must be 1..64")
    study, manifest = args.study.resolve(), args.manifest.resolve()
    logger = logger_for(study)
    lock = acquire_lock(study)
    try:
        roster = extend_roster(study, manifest)
        paper_ids = [str(row["paper_id"]) for row in roster]
        claims, reviews = completion(study, paper_ids)
        logger.info(
            "[准备] roster=%d claims=%d reviews=%d workers=%d cli_limit=%d",
            len(roster),
            claims,
            reviews,
            args.workers,
            args.cli_limit,
        )
        logger.info(
            "[模型] 新请求：model=%s reasoning_effort=%s service_tier=%s；"
            "已完成文件保留",
            RUN_MODEL,
            RUN_REASONING_EFFORT,
            RUN_SERVICE_TIER,
        )
        if args.dry_run:
            return
        env = stage_environment()
        for round_number in range(1, args.max_rounds + 1):
            if claims == reviews == 1_000:
                break
            started = time.monotonic()
            logger.info("[第 %d 轮] 并行启动 claims 与 reviews", round_number)
            run_round(study, args.workers, args.cli_limit, env)
            claims, reviews = completion(study, paper_ids)
            logger.info(
                "[第 %d 轮完成] claims=%d reviews=%d elapsed=%.1fs",
                round_number,
                claims,
                reviews,
                time.monotonic() - started,
            )
        if claims != 1_000 or reviews != 1_000:
            raise RuntimeError(
                f"incomplete after retries: claims={claims}, reviews={reviews}"
            )
        logger.info("[完成] 1000/1000 claims；1000/1000 reviews")
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[停止] 已终止 claims/reviews 子进程；已完成文件保留，可直接续跑。")
        raise SystemExit(130) from None

"""Small shared utilities; stages communicate only through ordinary JSON files."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from gear.config import GearConfig, load_config
from gear.env import getenv_runtime

T = TypeVar("T")


def setup_stage_logging(
    study: Path, stage: str, verbose: bool = False
) -> logging.Logger:
    """Write the same timestamped stage log to stdout and an append-only text file."""
    log_path = study / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"innovation_200.{stage}")
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler, level in (
        (logging.StreamHandler(sys.stdout), logging.DEBUG if verbose else logging.INFO),
        (logging.FileHandler(log_path, encoding="utf-8", mode="a"), logging.DEBUG),
    ):
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    else:
        payload = value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def experiment_config() -> GearConfig:
    """Default to Luna roles; child runners may explicitly override model/effort."""
    model = getenv_runtime("GEAR_STUDY_MODEL") or "gpt-5.6-luna"
    effort = getenv_runtime("GEAR_STUDY_REASONING_EFFORT")
    role_efforts = {
        "field_classifier": "low",
        "reference_extract": "low",
        "reference_check": "low",
        "graph_claim": "low",
        "claim_miner": "low",
        "supervisor_planner": "low",
        "claim_consolidator": "low",
        "internal_verifier": "high",
        "relation_fusion": "high",
        "graph_analysis": "high",
        "evaluation_judge": "high",
        "report_writer": "high",
        "pairwise_judge": "high",
    }
    return load_config(
        overrides={
            "model_cache_enabled": False,
            "relation_stability_check_enabled": False,
            "resume_fingerprint_checks_enabled": False,
            "role_model_override": model,
            "role_effort_overrides": {
                role: effort or default for role, default in role_efforts.items()
            },
            "max_claims": 8,
            "codex_cli": {"model": model, "reasoning_effort": effort or "high"},
            "retrieval": {
                "query_reasoning_effort": "low",
                "fulltext_max": 10,
                "relation_cards_max": 10,
                "retained_candidates_per_claim": 5,
                "minimum_comparable_candidates": 5,
            },
        }
    )


def configure_limits(cli_limit: int, relation_workers: int = 8) -> None:
    if not 1 <= cli_limit <= 64:
        raise ValueError("CLI concurrency must be between 1 and 64")
    network_limit = int(os.environ.get("GEAR_NETWORK_MAX_PROCESSES", "16"))
    if not 1 <= network_limit <= 32:
        raise ValueError("Network concurrency must be between 1 and 32")
    os.environ["GEAR_CLI_MAX_PROCESSES"] = str(cli_limit)
    os.environ["GEAR_NETWORK_MAX_PROCESSES"] = str(network_limit)
    os.environ.setdefault("GEAR_RELATION_WORKERS", str(relation_workers))
    os.environ.setdefault("GEAR_GPU_MAX_PROCESSES", "2")
    os.environ.setdefault("GEAR_MODEL_RETRIES", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")


def run_stage(
    rows: list[dict],
    worker: Callable[[dict], T],
    *,
    workers: int,
    status_path: Path,
    usage_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> list[dict]:
    """Run independent paper tasks; worker itself decides whether output already exists."""
    records: list[dict] = []
    stage_started = time.monotonic()
    if logger:
        logger.info(
            "[开始] 任务=%d，并发=%d，状态文件=%s",
            len(rows),
            workers,
            status_path,
        )

    def timed(row: dict) -> tuple[T, float]:
        from gear.innovation.usage import usage_log

        started = time.monotonic()
        task_id = str(row.get("task_id") or row["paper_id"])
        if usage_dir is None:
            return worker(row), time.monotonic() - started
        with usage_log(usage_dir / f"{task_id}.jsonl"):
            return worker(row), time.monotonic() - started

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(timed, row): str(row.get("task_id") or row["paper_id"])
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), 1):
            paper_id = futures[future]
            try:
                result, elapsed = future.result()
                record = {
                    "paper_id": paper_id,
                    "status": "complete",
                    "elapsed_seconds": elapsed,
                    "result": result,
                }
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                record = {
                    "paper_id": paper_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            records.append(record)
            write_json(status_path, records)
            elapsed = max(time.monotonic() - stage_started, 0.001)
            complete = sum(row["status"] == "complete" for row in records)
            failed = sum(row["status"] == "failed" for row in records)
            skipped = sum(
                row["status"] == "complete"
                and isinstance(row.get("result"), dict)
                and bool(row["result"].get("skipped"))
                for row in records
            )
            if logger:
                progress_log = (
                    logger.error if record["status"] == "failed" else logger.info
                )
                progress_log(
                    "[进度 %d/%d] %s：%s；完成=%d，跳过=%d，失败=%d，速度=%.2f任务/秒",
                    completed,
                    len(rows),
                    paper_id,
                    record["status"],
                    complete,
                    skipped,
                    failed,
                    completed / elapsed,
                )
                logger.debug("[任务详情] %s", json.dumps(record, ensure_ascii=False))
            else:
                print(json.dumps(record, ensure_ascii=False), flush=True)
    if logger:
        failed_count = sum(row["status"] == "failed" for row in records)
        completion_log = logger.warning if failed_count else logger.info
        completion_log(
            "[完成] 总任务=%d，成功=%d，跳过=%d，失败=%d，总耗时=%.1f秒",
            len(rows),
            sum(row["status"] == "complete" for row in records),
            sum(
                row["status"] == "complete"
                and isinstance(row.get("result"), dict)
                and bool(row["result"].get("skipped"))
                for row in records
            ),
            failed_count,
            time.monotonic() - stage_started,
        )
    return records


def generate_with_retries(call: Callable[[], T], retries: int = 2) -> T:
    for attempt in range(retries + 1):
        try:
            return call()
        except (OSError, RuntimeError, ValueError) as exc:
            if "limited access to this content" in str(exc) or attempt == retries:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("unreachable")


def wait_for_inputs(
    paths: list[Path],
    *,
    paper_id: str,
    producer_statuses: list[Path],
    logger: logging.Logger | None = None,
) -> None:
    """Wait for upstream files while independently launched stages stream results."""
    started = time.monotonic()
    last_report = -60.0
    deadline = time.monotonic() + float(
        os.environ.get("GEAR_STAGE_WAIT_TIMEOUT_SECONDS", "604800")
    )
    while not all(path.exists() for path in paths):
        waited = time.monotonic() - started
        if logger and waited - last_report >= 60.0:
            missing = [str(path) for path in paths if not path.exists()]
            logger.info(
                "[等待上游] %s：已等待=%.0f秒，缺少=%d；%s",
                paper_id,
                waited,
                len(missing),
                "；".join(missing),
            )
            last_report = waited
        for status_path in producer_statuses:
            try:
                records = read_json(status_path) if status_path.exists() else []
            except (OSError, json.JSONDecodeError):
                records = []
            if isinstance(records, list) and any(
                str(record.get("paper_id")) == paper_id
                and record.get("status") == "failed"
                for record in records
                if isinstance(record, dict)
            ):
                raise RuntimeError(
                    f"Upstream stage {status_path.stem} failed for {paper_id}"
                )
        if time.monotonic() >= deadline:
            missing = [str(path) for path in paths if not path.exists()]
            raise TimeoutError(f"Timed out waiting for inputs: {missing}")
        time.sleep(2)
    if logger and time.monotonic() - started >= 1.0:
        logger.info(
            "[上游就绪] %s：等待耗时=%.1f秒",
            paper_id,
            time.monotonic() - started,
        )

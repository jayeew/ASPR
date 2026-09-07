#!/usr/bin/env python3
"""Resume development acceptance, then execute the authorized frozen study."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gear.artifacts import read_jsonl, write_json
from gear.env import subprocess_environment
from gear.innovation.validation import validate_run

DEVELOPMENT = ("s41467-026-68293-8", "s41467-026-68313-7", "s41467-026-68382-8")


def command(parts: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(
            parts,
            cwd=ROOT,
            stdout=handle,
            stderr=handle,
            check=True,
            env=subprocess_environment(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--screen-pid", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    logs = output / "orchestration"
    logs.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GEAR_GPU_MAX_PROCESSES", "2")
    os.environ.setdefault("GEAR_RELATION_WORKERS", "8")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")

    def status(stage: str, **details: object) -> None:
        value = {"stage": stage, "timestamp": time.time(), **details}
        write_json(logs / "status.json", value)
        print(json.dumps(value), flush=True)

    def pilot(paper_id: str) -> dict:
        root = ROOT / "outputs/innovation_v2_development" / paper_id
        for attempt in range(3):
            command(
                [
                    sys.executable,
                    "-m",
                    "gear",
                    "review",
                    "--input-contract",
                    str(
                        ROOT
                        / "outputs/nature_2026_random3"
                        / paper_id
                        / "innovation_input.json"
                    ),
                    "--output-dir",
                    str(root),
                    "--stage",
                    "all",
                ],
                logs / f"{paper_id}.log",
            )
            result = validate_run(root)
            write_json(logs / f"{paper_id}_validation.json", result)
            if result["valid"]:
                return {"paper_id": paper_id, "attempt": attempt + 1, **result}
        raise RuntimeError(f"Development acceptance failed: {paper_id}")

    runner = [sys.executable, "scripts/innovation_experiment.py"]
    try:
        status("waiting_for_existing_jobs", pids=args.wait_pid)
        while any(Path(f"/proc/{pid}/cmdline").exists() for pid in args.wait_pid):
            time.sleep(20)
        status("development_acceptance")
        with ThreadPoolExecutor(max_workers=2) as pool:
            validations = list(pool.map(pilot, DEVELOPMENT))
        write_json(logs / "development_acceptance.json", validations)
        status("screening_completion")
        while args.screen_pid and Path(f"/proc/{args.screen_pid}/cmdline").exists():
            time.sleep(20)
        for attempt in range(4):
            incomplete = []
            for row in read_jsonl(output / "manifest.jsonl"):
                path = output / "references" / str(row["article_id"]) / "reference.json"
                if (
                    not path.exists()
                    or json.loads(path.read_text())["status"] != "complete"
                ):
                    incomplete.append(str(row["article_id"]))
            if not incomplete:
                break
            if attempt == 3:
                raise RuntimeError(
                    "Screening still incomplete after three retry passes"
                )
            parts = runner + ["screen", "--output", str(output), "--workers", "16"]
            for paper_id in incomplete:
                parts.extend(["--paper-id", paper_id])
            command(parts, logs / "screen_retries.log")
        else:
            raise RuntimeError("Screening still incomplete after three retry passes")
        status("freeze")
        if not (output / "protocol.json").exists():
            command(runner + ["freeze", "--output", str(output)], logs / "freeze.log")
        protocol = json.loads((output / "protocol.json").read_text())
        for split in ("development", "test"):
            status(f"running_{split}", papers=len(protocol[split]))
            command(
                runner
                + [
                    "run",
                    "--output",
                    str(output),
                    "--split",
                    split,
                    "--paper-workers",
                    "2",
                ],
                logs / f"{split}.log",
            )
            failed = []
            for paper_id in protocol[split]:
                path = output / "runs" / paper_id / "execution.json"
                if (
                    not path.exists()
                    or json.loads(path.read_text())["status"] != "finished"
                ):
                    failed.append(paper_id)
            if failed:
                raise RuntimeError(f"{split} execution failures: {failed}")
            command(
                runner + ["summarize", "--output", str(output), "--split", split],
                logs / f"{split}_summary.log",
            )
        status("diagnostics")
        command(
            runner + ["diagnostics", "--output", str(output)], logs / "diagnostics.log"
        )
        status("execution_finished_pending_scientific_acceptance")
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        status("stopped_on_error", error=str(exc))
        raise


if __name__ == "__main__":
    main()

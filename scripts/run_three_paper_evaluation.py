#!/usr/bin/env python3
"""Complete the eight-system evaluation for exactly the three accepted pilots."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gear.artifacts import read_jsonl, write_json
from gear.config import GearConfig, load_config
from gear.innovation.diagnostics import run_diagnostics
from gear.innovation.evaluation import aggregate, paired_comparisons
from gear.trace import sha256_file
from innovation_experiment import DEVELOPMENT, execute_one, sources

OUT = ROOT / "outputs/innovation_v2_three_paper_evaluation"
PILOT = ROOT / "outputs/innovation_v2_development"
STUDY = ROOT / "outputs/innovation_v2_study"


def prepare() -> tuple[list[dict], GearConfig]:
    load_config()  # Load the configured environment before passing resolved config.
    config = GearConfig.model_validate_json(
        (PILOT / sorted(DEVELOPMENT)[0] / "resolved_config.json").read_text()
    )
    rows = [
        r
        for r in read_jsonl(STUDY / "manifest.jsonl")
        if r["article_id"] in DEVELOPMENT
    ]
    if {r["article_id"] for r in rows} != DEVELOPMENT:
        raise ValueError("Three-paper manifest mismatch")
    for row in rows:
        pid = row["article_id"]
        source = PILOT / pid
        if (
            GearConfig.model_validate_json(
                (source / "resolved_config.json").read_text()
            )
            != config
        ):
            raise ValueError("Pilot configurations differ")
        blind = OUT / "runs" / pid / "blind"
        blind.mkdir(parents=True, exist_ok=True)
        for branch in ("shared", "gear", "graph", "fusion"):
            if not (blind / branch).exists():
                shutil.copytree(source / branch, blind / branch)
        for name in ("innovation_input.json", "resolved_config.json"):
            if not (blind / name).exists():
                shutil.copyfile(source / name, blind / name)
        input_dir = OUT / "inputs" / pid
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "innovation_input.json", input_dir / "input.json")
        refs = OUT / "audited_references" / pid
        if not refs.exists():
            shutil.copytree(STUDY / "audited_references" / pid, refs)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    protocol = {
        "scope": "Exactly three accepted pilots; descriptive evaluation, no held-out superiority claim",
        "papers": sorted(DEVELOPMENT),
        "blind_systems": [
            "direct",
            "direct_end_to_end",
            "rag",
            "gear",
            "graph",
            "fusion",
            "fusion_text_only",
            "fusion_no_metrics",
        ],
        "specified_systems": [
            "direct",
            "rag",
            "gear",
            "graph",
            "fusion",
            "fusion_text_only",
            "fusion_no_metrics",
        ],
        "no_reference_policy": "Human metrics null; specified task not applicable; retain blind diagnostics",
        "reuse": "Accepted GEAR/Graph/fusion and shared claims; fresh baseline/ablation directories",
        "diagnostics": "One independent fixed-evidence interpretation repeat, second judge, source checks, community intervention",
        "source_hashes": sources(),
        "runner_hash": sha256_file(Path(__file__)),
        "reference_hashes": {
            str(p.relative_to(OUT)): sha256_file(p)
            for p in (OUT / "audited_references").glob("*/reference.json")
        },
    }
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Three-paper protocol changed; inspect before resuming")
    write_json(path, protocol)
    return rows, config


def one(row: dict, config: GearConfig) -> dict:
    args = argparse.Namespace(output=OUT)
    status: dict = {}
    for attempt in range(3):
        status = execute_one(row, args, config)
        write_json(OUT / "attempts" / row["article_id"] / f"{attempt}.json", status)
        if status[
            "status"
        ] == "finished" or "limited access to this content for safety reasons" in str(
            status
        ):
            break
    if status["status"] != "finished":
        return status
    pid = row["article_id"]
    for task in ("blind", "specified"):
        root = OUT / "runs" / pid / task
        if (root / "not_applicable.json").exists():
            continue
        try:
            run_diagnostics(
                config,
                root,
                OUT / "audited_references" / pid / "reference.json",
                OUT / "diagnostics" / task / pid,
                task=task,
            )
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            status = {**status, "status": "diagnostics_failed", "error": str(exc)}
            break
    write_json(OUT / "runs" / pid / "completion.json", status)
    return status


def main() -> None:
    rows, config = prepare()
    records = []
    write_json(
        OUT / "status.json", {"status": "running", "papers": sorted(DEVELOPMENT)}
    )
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {pool.submit(one, row, config): row["article_id"] for row in rows}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "paper_id": futures[future],
                    "status": "failed",
                    "error": str(exc),
                }
            records.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            write_json(OUT / "status.json", {"status": "running", "outcomes": records})
    summaries = {}
    for subset in ("all", "strict"):
        scores = [
            json.loads(p.read_text())
            for p in (OUT / "scores").glob(f"*/*/*_{subset}.json")
        ]
        summaries[subset] = {
            "metrics": aggregate(scores),
            "paired_comparisons": paired_comparisons(scores),
            "score_files": len(scores),
        }
    write_json(OUT / "summary.json", summaries)
    write_json(
        OUT / "status.json",
        {
            "status": (
                "complete"
                if all(r["status"] == "finished" for r in records)
                else "incomplete"
            ),
            "outcomes": records,
        },
    )


if __name__ == "__main__":
    os.environ.setdefault("GEAR_GPU_MAX_PROCESSES", "2")
    os.environ.setdefault("GEAR_RELATION_WORKERS", "8")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    main()

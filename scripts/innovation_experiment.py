#!/usr/bin/env python3
"""Screen, freeze, execute and summarize the v2 study without overwriting old runs."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextvars import copy_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gear.artifacts import read_jsonl, read_model, write_json, write_model
from gear.config import GearConfig, load_config
from gear.innovation.contracts import AnalysisResult, ReferenceSet
from gear.innovation.evaluation import aggregate, evaluate, paired_comparisons
from gear.innovation.experiments import SYSTEMS, run_control, run_independent_baseline
from gear.innovation.references import screen
from gear.innovation.usage import usage_log
from gear.paper_extraction import prepare_input
from gear.review_contracts import BranchStatus, InnovationPaperInput
from gear.review_pipeline import review_paper
from gear.trace import sha256_file

SEED = 20260906
DEVELOPMENT = {"s41467-026-68293-8", "s41467-026-68313-7", "s41467-026-68382-8"}


def sources() -> dict[str, str]:
    paths = list((ROOT / "gear").rglob("*.py")) + [
        Path(__file__),
        ROOT / "scripts/advance_innovation_study.py",
        ROOT / "configs/gear/default.json",
    ]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def screen_data(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.manifest)
    if args.paper_id:
        rows = [x for x in rows if x["article_id"] in args.paper_id]
    config = load_config(args.config)

    def one(row: dict) -> dict:
        result = screen(
            str(row["article_id"]),
            Path(row["peer_review_markdown_path"]),
            Path(row["paper_markdown_path"]),
            args.output / "references" / str(row["article_id"]),
            config,
        )
        return {
            "paper_id": row["article_id"],
            "status": result.status.value,
            "retained": len(result.retained_ids),
            "limitations": result.limitations,
        }

    outcomes = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, row): row["article_id"] for row in rows}
        for future in as_completed(futures):
            try:
                record = future.result()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                record = {
                    "paper_id": futures[future],
                    "status": "failed",
                    "error": str(exc),
                }
            outcomes.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            write_json(args.output / "screening_status.json", outcomes)


def reference_quality(args: argparse.Namespace) -> None:
    from gear.innovation.reference_quality import audit_reference

    config = load_config(args.config)

    def one(row: dict) -> dict:
        paper_id = str(row["article_id"])
        source = args.output / "references" / paper_id
        target = args.output / "audited_references" / paper_id
        if (target / "unavailable.json").exists():
            return {"paper_id": paper_id, "status": "provider_unavailable"}
        original = read_model(source / "reference.json", ReferenceSet)
        try:
            if original.status is not BranchStatus.COMPLETE:
                raise RuntimeError("; ".join(original.limitations))
            result = audit_reference(config, source, target)
            return {
                "paper_id": paper_id,
                "status": "complete",
                "retained": len(result.retained_ids),
            }
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            write_json(target / "failure.json", {"error": str(exc)})
            if "limited access to this content for safety reasons" in str(exc):
                write_json(
                    target / "unavailable.json",
                    {
                        "paper_id": paper_id,
                        "reason": "provider_content_restriction",
                        "stage": "reference_quality",
                    },
                )
                return {"paper_id": paper_id, "status": "provider_unavailable"}
            return {"paper_id": paper_id, "status": "failed", "error": str(exc)}

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(one, read_jsonl(args.manifest)):
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            write_json(args.output / "reference_quality_status.json", records)
    if any(r["status"] == "failed" for r in records):
        raise RuntimeError(
            "Reference quality checks incomplete; inspect saved failures"
        )


def prepare_documents(args: argparse.Namespace) -> None:
    from datetime import date
    from shutil import copyfile

    def one(row: dict) -> dict:
        paper_id = str(row["article_id"])
        root = args.output / "inputs" / paper_id
        target = root / "input.json"
        if target.exists():
            return {"paper_id": paper_id, "status": "prepared"}
        root.mkdir(parents=True, exist_ok=True)
        copyfile(row["paper_markdown_path"], root / "paper.md")
        copyfile(row["peer_review_markdown_path"], root / "review.md")
        published = date.fromisoformat(str(row["publication_date"])[:10])
        for attempt in range(4):
            try:
                item = prepare_input(
                    paper_path=root / "paper.md",
                    paper_id=paper_id,
                    title=str(row["title"]),
                    doi=str(row["doi"]),
                    publication_date=published,
                    cutoff_date=published,
                    venue=str(row["journal_name"]),
                )
                break
            except OSError as exc:
                if attempt == 3:
                    return {"paper_id": paper_id, "status": "failed", "error": str(exc)}
                time.sleep(attempt + 1)
        write_model(target, item)
        return {"paper_id": paper_id, "status": "prepared"}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(one, read_jsonl(args.manifest)))
    write_json(args.output / "prepared_documents.json", records)
    print(
        json.dumps(
            {
                "prepared": sum(x["status"] == "prepared" for x in records),
                "failed": sum(x["status"] == "failed" for x in records),
            }
        ),
        flush=True,
    )
    if any(x["status"] == "failed" for x in records):
        raise RuntimeError(
            "Input preparation incomplete; retry resumes completed inputs"
        )


def freeze(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.manifest)
    groups: dict[tuple[str, str, str], list[str]] = {}
    excluded = []
    unavailable = []
    for row in rows:
        paper_id = str(row["article_id"])
        quality_root = args.output / "audited_references" / paper_id
        if (quality_root / "unavailable.json").exists():
            unavailable.append(paper_id)
            continue
        path = quality_root / "reference.json"
        if not path.exists():
            raise ValueError(f"Screening incomplete: {paper_id}")
        ref = read_model(path, ReferenceSet)
        if ref.status is not BranchStatus.COMPLETE:
            raise ValueError(f"Screening failed: {paper_id}")
        points = [
            x
            for x in ref.points
            if x.reference_id in ref.retained_ids and x.tier != "excluded"
        ]
        if not points:
            excluded.append(paper_id)
            continue
        tier = "A" if any(x.tier == "A" for x in points) else "B"
        stance = (
            "mixed"
            if len({x.stance for x in points if x.stance}) > 1
            else str(next((x.stance.value for x in points if x.stance), "none"))
        )
        groups.setdefault((str(row["journal_name"]), tier, stance), []).append(paper_id)
    rng = random.Random(SEED)
    dev = set(DEVELOPMENT) - set(excluded) - set(unavailable)
    test = []
    for key in sorted(groups):
        ids = sorted(set(groups[key]) - DEVELOPMENT)
        rng.shuffle(ids)
        n = round(len(ids) * 0.2)
        dev.update(ids[:n])
        test.extend(ids[n:])
    protocol = {
        "schema_version": "innovation_v2",
        "seed": SEED,
        "source_hashes": sources(),
        "graph_hashes": {
            str(p.relative_to(ROOT)): sha256_file(p)
            for p in sorted((ROOT / "data/claim_graph").glob("*"))
            if p.is_file() and p.suffix in (".sqlite", ".parquet", ".npy", ".faiss")
        },
        "manifest_hash": sha256_file(args.manifest),
        "input_hashes": {
            str(p.relative_to(args.output)): sha256_file(p)
            for p in sorted((args.output / "inputs").glob("*/*"))
            if p.is_file()
        },
        "quality_hashes": {
            str(p.relative_to(args.output)): sha256_file(p)
            for p in sorted((args.output / "audited_references").glob("*/*.json"))
            if p.name != "failure.json"
        },
        "reference_hashes": {
            str(p.relative_to(args.output)): sha256_file(p)
            for p in sorted(
                (args.output / "audited_references").glob("*/reference.json")
            )
        },
        "development": sorted(dev),
        "test": sorted(test),
        "excluded": excluded,
        "provider_unavailable": unavailable,
        "reference_root": "audited_references",
        "reference_definition": "Source-only audited explicit contribution/novelty opinions; model-assisted, not expert gold",
        "replicate_definition": "Two new interpretation generations with distinct replicate IDs, fixed evidence; retrieval stability is not inferred",
        "regression_only": sorted(DEVELOPMENT & set(excluded)),
        "diagnostic_papers": sorted(
            random.Random(SEED).sample(
                sorted(test), min(len(test), max(10, round(len(test) * 0.1)))
            )
        ),
        "systems": [*SYSTEMS, "direct_end_to_end"],
        "metrics": [
            "coverage",
            "matched_stance_agreement",
            "found_and_agreed",
            "reason_coverage",
            "reason_contradiction_rate",
        ],
        "tasks": ["blind", "specified"],
        "config": load_config(args.config).model_dump(mode="json"),
    }
    for paper_id in sorted(dev | set(test)):
        if not (args.output / "inputs" / paper_id / "input.json").exists():
            raise ValueError(f"Input snapshot missing: {paper_id}")
    target = args.output / "protocol.json"
    if target.exists():
        raise FileExistsError("Protocol already frozen; choose a new study directory")
    write_json(target, protocol)
    print(
        json.dumps(
            {"development": len(dev), "test": len(test), "excluded": len(excluded)}
        )
    )


def execute(args: argparse.Namespace) -> None:
    protocol = json.loads((args.output / "protocol.json").read_text())
    if protocol["source_hashes"] != sources():
        raise ValueError("Code changed after freeze")
    for path, digest in protocol["graph_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("Graph assets changed after freeze")
    if protocol["manifest_hash"] != sha256_file(args.manifest):
        raise ValueError("Manifest changed after freeze")
    for path, digest in {
        **protocol["reference_hashes"],
        **protocol["input_hashes"],
        **protocol["quality_hashes"],
    }.items():
        if sha256_file(args.output / path) != digest:
            raise ValueError("References changed after freeze")
    config = load_config(args.config)
    if config.model_dump(mode="json") != protocol["config"]:
        raise ValueError("Config changed after freeze")
    ids = set(protocol[args.split])
    if args.paper_id:
        if not set(args.paper_id).issubset(ids):
            raise ValueError("Requested paper outside frozen split")
        ids.intersection_update(args.paper_id)
    rows = [x for x in read_jsonl(args.manifest) if x["article_id"] in ids]
    os.environ.setdefault("GEAR_GPU_MAX_PROCESSES", str(args.paper_workers))
    os.environ.setdefault("GEAR_RELATION_WORKERS", "8")
    # Spawn avoids inheriting CUDA state; each paper owns its evidence directories.
    with ProcessPoolExecutor(
        max_workers=args.paper_workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = {
            pool.submit(execute_one, row, args, config): row["article_id"]
            for row in rows
        }
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


def execute_one(row: dict, args: argparse.Namespace, config: GearConfig) -> dict:
    paper_id = str(row["article_id"])
    root = args.output / "runs" / paper_id
    started = time.monotonic()
    try:
        item = read_model(
            args.output / "inputs" / paper_id / "input.json", InnovationPaperInput
        )
        reference = read_model(
            args.output / "audited_references" / paper_id / "reference.json",
            ReferenceSet,
        )
        selected = [
            x
            for x in reference.points
            if x.reference_id in reference.retained_ids and x.tier != "excluded"
        ]
        grouped = {x.contribution_id: x.target_text for x in selected}
        with usage_log(root / "model_usage.jsonl"):
            for task in ("blind", "specified"):
                task_root = root / task
                if task == "specified" and not grouped:
                    write_json(
                        task_root / "not_applicable.json",
                        {"reason": "No eligible human contribution targets"},
                    )
                    continue
                targets = list(grouped.values()) if task == "specified" else None
                review_paper(
                    item, output_dir=task_root, config=config, target_claims=targets
                )
                modes = (*SYSTEMS, "direct_end_to_end") if task == "blind" else SYSTEMS

                def score_mode(
                    mode: str, task_root: Path = task_root, task: str = task
                ) -> None:
                    result = (
                        read_model(task_root / mode / "analysis.json", AnalysisResult)
                        if mode in ("gear", "graph", "fusion")
                        else (
                            run_independent_baseline(item, task_root, config)
                            if mode == "direct_end_to_end"
                            else run_control(item, task_root, config, mode)
                        )
                    )
                    for uncertain in (False, True):
                        score_path = (
                            args.output
                            / "scores"
                            / task
                            / paper_id
                            / f'{mode}_{"strict" if uncertain else "all"}.json'
                        )
                        for attempt in range(3):
                            scored = evaluate(
                                config,
                                result,
                                reference,
                                (
                                    score_path
                                    if attempt == 0
                                    else score_path.with_name(
                                        f"{score_path.stem}_retry{attempt}.json"
                                    )
                                ),
                                exclude_uncertain=uncertain,
                                task=task,
                                nonce=f"format-retry-{attempt}" if attempt else "",
                            )
                            if not scored["failures"]:
                                write_json(score_path, scored)
                                break
                            if (
                                "limited access to this content for safety reasons"
                                in str(scored["failures"])
                            ):
                                break
                        if scored["failures"]:
                            raise RuntimeError(
                                f"Evaluation failed: {paper_id}/{task}/{mode}"
                            )

                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [
                        pool.submit(copy_context().run, score_mode, mode)
                        for mode in modes
                    ]
                    for future in futures:
                        future.result()
        status = {
            "paper_id": paper_id,
            "status": "finished",
            "seconds": time.monotonic() - started,
        }
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        status = {
            "paper_id": paper_id,
            "status": "failed",
            "error": str(exc),
            "seconds": time.monotonic() - started,
        }
    write_json(root / "execution.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "screen",
            "quality",
            "prepare",
            "freeze",
            "run",
            "diagnostics",
            "summarize",
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--paper-id", action="append")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--paper-workers", type=int, default=2)
    parser.add_argument(
        "--split", choices=("development", "test"), default="development"
    )
    args = parser.parse_args()
    if args.workers < 1 or args.paper_workers < 1:
        parser.error("Worker counts must be positive")
    if args.manifest is None:
        args.manifest = args.output / "manifest.jsonl"
        if not args.manifest.exists():
            if args.command != "screen":
                raise ValueError("Screening manifest snapshot missing")
            from shutil import copyfile

            args.output.mkdir(parents=True, exist_ok=True)
            copyfile(ROOT / "data/nature_2026_testset/manifest.jsonl", args.manifest)
    if args.command == "screen":
        screen_data(args)
    elif args.command == "quality":
        reference_quality(args)
    elif args.command == "prepare":
        prepare_documents(args)
    elif args.command == "freeze":
        freeze(args)
    elif args.command == "run":
        execute(args)
    elif args.command == "diagnostics":
        from gear.innovation.diagnostics import run_diagnostics

        protocol = json.loads((args.output / "protocol.json").read_text())
        if protocol["source_hashes"] != sources():
            raise ValueError("Code changed after freeze")
        for paper_id in protocol["diagnostic_papers"]:
            for task in ("blind", "specified"):
                run_diagnostics(
                    load_config(args.config),
                    args.output / "runs" / paper_id / task,
                    args.output / "audited_references" / paper_id / "reference.json",
                    args.output / "diagnostics" / task / paper_id,
                    task=task,
                )
    else:
        protocol = json.loads((args.output / "protocol.json").read_text())
        selected = set(protocol[args.split])
        rows = [
            json.loads(p.read_text())
            for p in (args.output / "scores").glob("*/*/*_all.json")
            if p.parent.name in selected
        ]
        write_json(
            args.output / f"summary_{args.split}.json",
            {
                "split": args.split,
                "metrics": aggregate(rows),
                "paired_comparisons": paired_comparisons(rows),
            },
        )


if __name__ == "__main__":
    main()

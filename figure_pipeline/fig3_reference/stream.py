"""Evaluate available reports continuously without duplicating live jobs."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
import os
import random
import time
from typing import Any

import psutil

from . import evaluate
from .settings import METHODS, OUTPUT, missing_path, roster, write


def unavailable(stage:str,task:Any) -> bool:
    if stage=="clusters":
        return missing_path("clusters",task).exists()
    ident,method=task[:2]
    if stage=="preference":
        return any(missing_path("reports",ident,m).exists() for m in ("fusion",method)) or missing_path(stage,ident,"_".join(task[1:])).exists()
    return missing_path("reports",ident,method).exists() or missing_path(stage,ident,method).exists() or (stage=="support" and missing_path("extract",ident,method).exists())


def ready(stage: str, task: Any) -> bool:
    if stage == "clusters":
        return all((OUTPUT / "data/support" / m / f"{task}.json").exists() or any(missing_path(s,task,m).exists() for s in ("reports","extract","support")) for m in METHODS)
    if stage == "preference":
        return all((OUTPUT / "data/reports" / m / f"{task[0]}.json").exists() for m in ("fusion",task[1]))
    ident, method = task
    folder = "units" if stage == "support" else "reports"
    return (OUTPUT / "data" / folder / method / f"{ident}.json").exists()


def destination(stage: str, task: Any) -> bool:
    if stage == "clusters":
        return (OUTPUT / "data/insight_clusters" / f"{task}.json").exists()
    if stage == "preference":
        return (OUTPUT / "data/preferences" / task[1] / f"{task[0]}_{task[2]}.json").exists()
    ident, method = task
    folder = "units" if stage == "extract" else stage
    return (OUTPUT / "data" / folder / method / f"{ident}.json").exists()


def run(stage: str, workers: int, skip: int) -> None:
    os.environ["GEAR_CLI_MAX_PROCESSES"] = os.environ.get("FIG3_CLI_LIMIT","80")
    tasks = [(p["paper_id"], m) for p in roster()[skip:] for m in METHODS]
    if stage == "clusters":
        tasks = [p["paper_id"] for p in roster()[skip:]]
    elif stage == "preference":
        tasks = [(p["paper_id"],m,o) for p in roster()[skip:] for m in METHODS[:-1] for o in ("AB","BA")]
        random.Random(20260917).shuffle(tasks)
    pending = [t for t in tasks if not destination(stage,t)]
    active: dict[Future, tuple[str,str]] = {}
    errors, finished = [], len(tasks) - len(pending)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            blocked=[t for t in pending if unavailable(stage,t)]
            for task in blocked:
                pending.remove(task)
                ident=task if isinstance(task,str) else task[0]
                method="" if isinstance(task,str) else "_".join(task[1:])
                write(missing_path(stage,ident,method),{"paper_id":ident,"method":method,"status":"not_assessed","reason":"unavailable_upstream_or_content_rejection"})
                finished+=1
            if psutil.virtual_memory().available >= 5 * 1024**3:
                available = [t for t in pending if ready(stage,t)][:workers-len(active)]
                for task in available:
                    pending.remove(task)
                    active[pool.submit(getattr(evaluate,stage),task)] = task
            if not active:
                time.sleep(3)
                continue
            done, _ = wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                try:
                    result = future.result()
                except (OSError,ValueError,RuntimeError,TimeoutError) as exc:
                    result = {"task":str(task),"error":str(exc)[-1800:]}
                    errors.append(result)
                    if "limited access to this content" in str(exc).lower():
                        ident=task if isinstance(task,str) else task[0]
                        method="" if isinstance(task,str) else "_".join(task[1:])
                        write(missing_path(stage,ident,method),{"paper_id":ident,"method":method,"status":"not_assessed","reason":"model_content_rejection"})
                finished += 1
                print(f"{stage} {finished}/{len(tasks)} {result}",flush=True)
            write(OUTPUT / "data" / f"{stage}_live_progress.json",{"finished":finished,"total":len(tasks),"active":len(active),"waiting":len(pending),"errors":errors})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("extract","quality","support","clusters","preference"))
    parser.add_argument("--workers",type=int,default=32)
    parser.add_argument("--skip-first",type=int,default=0)
    args = parser.parse_args()
    run(args.stage,args.workers,args.skip_first)


if __name__ == "__main__":
    main()

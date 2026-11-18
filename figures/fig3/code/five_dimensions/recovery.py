"""After the main batch finishes, recover blocked evaluations with one shared scheduler."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DAMAGE_ID, OUTPUT
from .io import read, write
from .models import Report
from .partial_reference import recover_reference
from .runner import Study, input_payload


def archive_status(output: Path, destination: Path) -> None:
    folder = destination / 'previous_status'
    folder.mkdir(parents=True, exist_ok=True)
    for path in (output / 'status').glob('*.json'):
        shutil.copy2(path, folder / path.name)


async def evaluate_recovered(study: Study, paper: str, method: str, envelope: dict[str, Any]) -> None:
    from .aggregate import current_request
    try:
        report = Report.model_validate(read(study.output / 'reports' / method / f'{paper}.json')).model_dump(mode='json')
        own = read(study.output / 'inputs/generation' / method / f'{paper}.json')
        current_request(study.output, 'generation', paper, method, input_payload(own, 'report'))
        await study.evaluate(paper, method, report, envelope['checklist'], partial_reference=envelope)
    except (ValueError, RuntimeError, OSError, TimeoutError, KeyError) as exc:
        study.status(f'recovery/{method}/{paper}', 'failed', error_type=type(exc).__name__, error=str(exc))


async def recover_async(output: Path, previous: dict[str, Any], maximum: int, initial: int,
                        timeout: int, selected: list[str] | None) -> dict[str, Any]:
    study, tasks = Study(output, maximum, initial, timeout), []
    roster = sorted(read(output / 'inputs/roster.json'), key=lambda row: row['paper_id'] != DAMAGE_ID)
    reference_states = previous.get('tasks', previous)
    for paper in roster:
        pid = paper['paper_id']
        if selected and pid not in selected:
            continue
        failed = reference_states.get(f'reference/{pid}', {}).get('state') == 'failed'
        if not failed or (output / 'references' / f'{pid}.json').exists():
            continue
        try:
            envelope = recover_reference(output, pid)
        except (ValueError, OSError, KeyError) as exc:
            study.status(f'recovery/reference/{pid}', 'failed', error_type=type(exc).__name__, error=str(exc))
            continue
        for method in paper['methods']:
            pipeline = reference_states.get(f'pipeline/{method}/{pid}', {})
            if pipeline.get('blocked_by') == f'reference/{pid}':
                tasks.append(evaluate_recovered(study, pid, method, envelope))
    progress = asyncio.create_task(study.progress())
    try:
        await asyncio.gather(*tasks)
    finally:
        progress.cancel()
        await asyncio.gather(progress, return_exceptions=True)
    return {'operation': 'recover_reference_blocked_evaluations', 'scheduled_evaluations': len(tasks),
            'scheduler': study.scheduler.snapshot(), 'tasks': study.records,
            'reference_generation_calls': 0, 'reference_location_calls': 0,
            'all_completion_preserved': True,
            'administrative_only_evaluations': sum(r['state'] == 'technical_failure_only' for r in study.records.values()),
            'failed_tasks': sum(r['state'] == 'failed' for r in study.records.values())}


def recover(output: Path = OUTPUT, maximum: int = 64, initial: int = 16,
            timeout: int = 900, selected: list[str] | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Main batch or another recovery is active; no second scheduler may start') from exc
        completion = output / 'status/all_completion.json'
        if not completion.is_file():
            raise ValueError('Main batch all_completion.json is required before reference recovery')
        previous = read(completion)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        destination = output / 'recovery_runs' / stamp
        archive_status(output, destination)
        result = asyncio.run(recover_async(output, previous, maximum, initial, timeout, selected))
        write(destination / 'recovery_completion.json', result)
        write(output / 'status/recovery_completion.json', {**result, 'run_archive': str(destination)})
        write(output / 'status/progress.json', {
            **result['scheduler'], 'state': 'finished', 'cohort': 'recovery',
            'tasks': dict(Counter(row['state'] for row in result['tasks'].values())),
        })
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--maximum', type=int, default=64)
    parser.add_argument('--initial', type=int, default=16)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--paper', action='append', dest='selected')
    arguments = parser.parse_args()
    result = recover(arguments.output, arguments.maximum, arguments.initial, arguments.timeout, arguments.selected)
    if result['failed_tasks']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

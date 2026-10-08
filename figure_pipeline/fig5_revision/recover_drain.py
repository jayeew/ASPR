"""Recover completed CLI responses after an explicitly recorded scheduler drain."""
from __future__ import annotations

import json
import os
import signal
from datetime import datetime
from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_100.models import Analysis, Report
from figure_pipeline.fig5_robustness.runtime import now

from .models import Config, Evaluation, Reference, Supports


def process_state(pid: int) -> str:
    path = Path(f'/proc/{pid}/stat')
    return path.read_text().split()[2] if path.exists() else 'gone'


def progress(config: Config) -> dict[str, Any]:
    drain = read(config.output / 'dispatch_drain.json')
    active = [r for r in drain['active_at_drain'] if r.get('pid') and process_state(r['pid']) not in ('gone', 'Z')]
    return {'total': len(drain['active_at_drain']), 'still_running': len(active), 'active': active}


def recover(config: Config) -> None:
    drain = read(config.output / 'dispatch_drain.json')
    if progress(config)['still_running']:
        raise RuntimeError('Already-dispatched CLIs are still executing; do not interrupt their responses')
    pid = drain['parent_pid']
    if process_state(pid) not in ('gone', 'Z'):
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
        if process_state(pid) != 'T' or b'figure_pipeline.fig5_revision' not in command:
            raise RuntimeError('Expected explicitly stopped Fig5 scheduler; refusing unrelated process')
        os.kill(pid, signal.SIGKILL)
    schemas = {'reference_final': Reference, 'supports': Supports, 'supports_v2': Supports,
               'analysis': Analysis, 'reports': Report, 'evaluation': Evaluation, 'independent_evaluation': Evaluation}
    results = []
    for entry in drain['active_at_drain']:
        path = Path(entry['record'])
        record = read(path)
        completed = False
        for line in path.with_name('events.jsonl').read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('type') == 'turn.completed':
                completed = True
                record['usage'] = event.get('usage')
            if event.get('type') in ('error', 'turn.failed'):
                record['cli_error'] = json.dumps(event, ensure_ascii=False)
        answer = path.with_name('answer.json')
        try:
            if not completed:
                raise ValueError('No turn.completed event; inspect original CLI failure')
            value = schemas[record['stage']].model_validate_json(answer.read_text()).model_dump()
            target = config.output / record['stage'] / record['method'] / f"{record['paper_id']}.json"
            if target.exists() and read(target) != value:
                raise ValueError('Completed stage differs from recovered response')
            write(path.with_name('response.json'), value)
            write(target, value)
            record['state'] = 'completed'
            record['seconds'] = max(0, answer.stat().st_mtime - datetime.fromisoformat(record['time']).timestamp())
            record['timing_source'] = 'recovered: request-record UTC to CLI answer-file mtime; excludes suspended supervisor delay'
        except (OSError, ValueError) as exc:
            record.update(state='failed', error=str(exc), error_type='DrainedResponseRecoveryError')
        record.update(recovered_after_dispatch_drain=True, recovery_time=now())
        write(path, record)
        write(config.output / 'task_status' / record['stage'] / record['method'] / f"{record['paper_id']}.json",
              {'paper_id': record['paper_id'], 'stage': record['stage'], 'condition': record['method'], 'state': record['state'],
               'time': now(), 'reason': 'Recovered finished CLI; semantic validation remains in normal pipeline'})
        results.append({'record': str(path), 'state': record['state']})
    drain.update(state='recovered_ready_to_resume', recovered=results, recovery_time=now())
    write(config.output / 'dispatch_drain.json', drain)
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    recover(Config())

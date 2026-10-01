"""Execute only the 681 human-approved first-stage CD requests."""
from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any

from .cd_contracts import EFFORTS, PROMPTS, SCHEMAS
from .cd_sources import OUTPUT, Locator, read, write_json
from ..fig3_revision.config import load_config
from ..fig3_revision.resources import admission

EXECUTION = OUTPUT / 'execution'


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def selected_packets() -> list[dict]:
    readiness = {r['id']: r for r in rows(OUTPUT/'tables/source_readiness.jsonl')}
    packets = [r for task in ('c_match', 'd_triage') for r in rows(OUTPUT/'packets'/f'{task}.jsonl')
               if readiness[r['id']]['ready_for_approval']]
    counts = Counter(r['task'] for r in packets)
    if len(packets) != 681 or counts != {'c_match': 420, 'd_triage': 261}:
        raise ValueError(f'Approved scope differs: {dict(counts)}')
    if len({r['id'] for r in packets}) != len(packets):
        raise ValueError('Duplicate task ID')
    if any(r['model'] != 'gpt-5.6-luna' for r in packets):
        raise ValueError('Model differs from approval')
    return packets


def result_path(packet: dict) -> Path:
    return EXECUTION/'results'/f'{packet["id"]}.json'


def semantic_integrity(packet: dict, answer: dict) -> list[str]:
    """Check provenance and identity only; do not decide scientific truth."""
    locator, issues = Locator(), []
    payload = packet['payload']
    if packet['task'] == 'c_match':
        cores = {r['core_id']: r for r in payload['cores']}
        expected = {(cid, payload['review']['concern_id']) for cid in cores}
        found = [(r['core_id'], r['concern_id']) for r in answer['links']]
        if set(found) != expected or len(found) != len(expected):
            issues.append('core_concern_identity_or_completeness')
        for row in answer['links']:
            core = cores.get(row['core_id'])
            if core is None:
                continue
            for field, text in [('manuscript_quotes', core['manuscript_quote']),
                                ('review_quotes', payload['review']['quote']),
                                ('reason_quotes', payload['review']['quote'])]:
                if any(locator.locate(text, q)['state'] not in {'exact', 'normalized_whitespace_unicode'}
                       for q in row[field]):
                    issues.append(f'{row["core_id"]}/{field}/not_located')
            if row['match'] == 'same' and (not row['manuscript_quotes'] or not row['review_quotes'] or
                    any(row[k] is not True for k in ('same_object', 'same_scope', 'explicit_novelty'))):
                issues.append(f'{row["core_id"]}/same_without_required_evidence')
    else:
        units = {r['unit_id']: r['quote'] for r in payload['statements']}
        supplied = json.dumps(payload, ensure_ascii=False)
        for i, part in enumerate(answer['parts']):
            if not part['unit_ids'] or any(uid not in units for uid in part['unit_ids']):
                issues.append(f'part{i}/unit_identity')
            texts = [units[uid] for uid in part['unit_ids'] if uid in units]
            if not part['report_quotes'] or any(not any(locator.locate(text, quote)['state'] in
                    {'exact', 'normalized_whitespace_unicode'} for text in texts) for quote in part['report_quotes']):
                issues.append(f'part{i}/report_quote_not_located')
            for field in ('subject_id', 'object_id'):
                if part[field] and part[field] not in supplied:
                    issues.append(f'part{i}/{field}/not_supplied')
    return issues


def execute(packet: dict, config: Any) -> dict:
    # Lazy import: preparing or inspecting packets cannot launch model calls.
    from ..fig3_revision.client import Client
    path = result_path(packet)
    if path.exists():
        return read(path)
    start = time.monotonic()
    client = Client(config, context={'paper_id': packet['payload']['paper_id'],
                                     'stage': 'cd_first_stage', 'step': packet['id']})
    record = {'id': packet['id'], 'task': packet['task'], 'model': 'gpt-5.6-luna',
              'effort': EFFORTS[packet['task']], 'status': 'failed', 'answer': None}
    try:
        answer = client.call(packet['task'], PROMPTS[packet['task']], packet['payload'],
                             SCHEMAS[packet['task']])
        issues = semantic_integrity(packet, answer)
        record.update(answer=answer, status='completed' if not issues else 'pending_integrity',
                      integrity_issues=issues)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        record.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        record['seconds'] = time.monotonic()-start
        write_json(path, record)
    return record


def main() -> None:
    packets = selected_packets()
    config = load_config().model_copy(update={'output': EXECUTION, 'model': 'gpt-5.6-luna',
        'efforts': {**EFFORTS}, 'workers': 64, 'cli_limit': 64, 'cli_initial': 16,
        'timeout_seconds': 900, 'synthesis_timeout_seconds': 900, 'repair_attempts': 1,
        'network_retries': 0, 'download_fulltext': False})
    write_json(EXECUTION/'authorization.json', {'source': 'Human user in this chat: 批准这681次第一阶段调用',
        'approved_tasks': {'c_match': 420, 'd_triage': 261}, 'initial_requests': 681,
        'maximum_calls_including_format_repair': 1362, 'model': 'gpt-5.6-luna',
        'automatic_failed_call_retries': 0, 'format_repair_limit': 1,
        'excluded': ['source_recovery', 'c_reason', 'd_verify', 'd_trace', 'retrieval', 'plotting']})
    pending = [p for p in packets if not result_path(p).exists()]
    statuses = Counter(read(result_path(p))['status'] for p in packets if result_path(p).exists())
    active: dict[Future, str] = {}
    position, ceiling, successful_since_ramp = 0, 16, 0
    last_print = 0.0
    with ThreadPoolExecutor(max_workers=64) as executor:
        while position < len(pending) or active:
            slots, memory, cpu_ceiling = admission(config, ceiling, len(active))
            for _ in range(min(slots, len(pending)-position)):
                packet = pending[position]
                active[executor.submit(execute, packet, config)] = packet['id']
                position += 1
            if active:
                done, _ = wait(active, timeout=2, return_when=FIRST_COMPLETED)
                for future in done:
                    ident = active.pop(future)
                    record = future.result()
                    statuses[record['status']] += 1
                    if record['status'] in {'completed', 'pending_integrity'}:
                        successful_since_ramp += 1
                    if successful_since_ramp >= 8 and ceiling < 64:
                        ceiling = min(64, ceiling+16)
                        successful_since_ramp = 0
            else:
                time.sleep(2)
            now = time.monotonic()
            if now-last_print >= 30 or (position == len(pending) and not active):
                print(json.dumps({'finished': sum(statuses.values()), 'total': 681,
                    'status': dict(statuses), 'active': len(active), 'concurrency_target': ceiling,
                    'cpu_ceiling': cpu_ceiling, 'memory_available_gib': round(memory, 1)}, ensure_ascii=False), flush=True)
                last_print = now
    write_json(EXECUTION/'completion.json', {'tasks': 681, 'statuses': dict(statuses),
                                           'all_tasks_attempted': True})


if __name__ == '__main__':
    main()

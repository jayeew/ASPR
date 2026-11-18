"""Execute the 52 separately approved source-bound D verification requests."""
from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from .cd_contracts import PROMPTS, SCHEMAS
from .cd_sources import OUTPUT, Locator, read, write_json
from .run_cd_first import rows
from ..fig3_revision.config import load_config
from ..fig3_revision.resources import admission

RUN = OUTPUT / 'd_verification'


def packets() -> list[dict]:
    selected = rows(OUTPUT/'first_stage/next_packets/d_verify.jsonl')
    if len(selected) != 52 or len({p['id'] for p in selected}) != 52:
        raise ValueError('Approved request population must remain 52')
    if any(p['model'] != 'gpt-5.6-luna' or p['effort'] != 'xhigh' or p['task'] != 'd_verify' for p in selected):
        raise ValueError('Approved model/task/effort differs')
    return selected


def path_for(packet: dict) -> Path:
    return RUN/'results'/f'{packet["id"]}.json'


def integrity(payload: dict, answer: dict) -> list[str]:
    issues = []
    if answer['relation_id'] != payload['relation_id']:
        issues.append('relation_identity')
    evidence = {e['evidence_id']: e for e in payload['original_evidence']}
    if any(eid not in evidence for eid in answer['evidence_ids']):
        issues.append('unsupplied_evidence_identity')
    texts = [evidence[eid]['quote'] for eid in answer['evidence_ids'] if eid in evidence]
    locator = Locator()
    if any(not any(locator.locate(text, quote)['state'] in {'exact','normalized_whitespace_unicode'}
                   for text in texts) for quote in answer['source_quotes']):
        issues.append('quote_not_in_cited_original_evidence')
    if answer['state'] != 'insufficient_material' and (not texts or not answer['source_quotes']):
        issues.append('asserted_verdict_without_locatable_evidence')
    if answer['state'] == 'supported_after_narrowing' and not answer['supported_statement']:
        issues.append('narrowed_relation_missing')
    return issues


def execute(packet: dict, config: object) -> dict:
    from ..fig3_revision.client import Client
    target = path_for(packet)
    if target.exists():
        return read(target)
    started = time.monotonic()
    paper = packet['payload']['relation_id'].split('/')[0]
    record = {'id': packet['id'], 'relation_id': packet['payload']['relation_id'],
              'model': 'gpt-5.6-luna', 'effort': 'xhigh', 'status': 'failed', 'answer': None}
    try:
        client = Client(config, context={'paper_id': paper, 'stage': 'd_verification', 'step': packet['id']})
        answer = client.call('d_verify', PROMPTS['d_verify'], packet['payload'], SCHEMAS['d_verify'])
        issues = integrity(packet['payload'], answer)
        record.update(answer=answer, status='completed' if not issues else 'pending_integrity', integrity_issues=issues)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        record.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        record['seconds'] = time.monotonic()-started
        write_json(target, record)
    return record


def main() -> None:
    selected = packets()
    config = load_config().model_copy(update={'output': RUN, 'model': 'gpt-5.6-luna',
        'efforts': {'d_verify': 'xhigh'}, 'workers': 64, 'cli_limit': 64, 'cli_initial': 16,
        'timeout_seconds': 900, 'synthesis_timeout_seconds': 900, 'repair_attempts': 1,
        'network_retries': 0, 'download_fulltext': False})
    write_json(RUN/'authorization.json', {'source': 'Human user approved 52 D original-evidence verifications in this chat.',
        'requests': 52, 'maximum_calls_with_format_repair': 104, 'model': 'gpt-5.6-luna',
        'effort': 'xhigh', 'automatic_failed_retries': 0,
        'excluded': ['d_trace','c_reason','source_recovery','retrieval','plotting']})
    pending = [p for p in selected if not path_for(p).exists()]
    states = Counter(read(path_for(p))['status'] for p in selected if path_for(p).exists())
    active: dict[Future, str] = {}
    position, ceiling, successes, last_print = 0, 16, 0, 0.0
    with ThreadPoolExecutor(max_workers=64) as executor:
        while position < len(pending) or active:
            slots, memory, _ = admission(config, ceiling, len(active))
            for _ in range(min(slots,len(pending)-position)):
                packet = pending[position]
                active[executor.submit(execute,packet,config)] = packet['id']
                position += 1
            if active:
                done, _ = wait(active,timeout=2,return_when=FIRST_COMPLETED)
                for future in done:
                    active.pop(future)
                    record = future.result()
                    states[record['status']]+=1
                    if record['status'] != 'failed':
                        successes += 1
                    if successes >= 8 and ceiling < 64:
                        ceiling = min(64,ceiling+16)
                        successes = 0
            else:
                time.sleep(2)
            now = time.monotonic()
            if now-last_print >= 30 or (position==len(pending) and not active):
                print(json.dumps({'finished':sum(states.values()),'total':52,'status':dict(states),
                    'active':len(active),'concurrency_target':ceiling,'memory_available_gib':round(memory,1)}),flush=True)
                last_print=now
    write_json(RUN/'completion.json',{'requests':52,'statuses':dict(states),'all_requests_attempted':True})


if __name__ == '__main__':
    main()

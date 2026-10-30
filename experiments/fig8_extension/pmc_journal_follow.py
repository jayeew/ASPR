"""Consume newly harvested PMC journal records without issuing network requests."""
from __future__ import annotations

import fcntl
import json
import time

from .derived_refresh import refresh
from .publisher_delta_apply import apply_available
from .springer_author_cache import run as apply_authors
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/pmc_journal_batches'


def record_attempts(receipts: list[str]) -> None:
    """Link each consumed batch outcome to its task; retrieval is not task closure."""
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        for receipt in receipts:
            record = json.loads((ROOT / receipt).read_text())
            state = record['status']
            if state == 'parsed' and record['doi'] not in record['parsed']['doi']:
                state = 'doi_identity_mismatch'
            for task_id, in db.execute(
                "SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?", (record['doi'],)
            ).fetchall():
                if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                              (task_id, 'pmc_journal_front_batch', receipt)).fetchone():
                    continue
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                           (task_id, 'pmc_journal_front_batch', state, receipt,
                            json.dumps({'batch_receipt': record.get('batch_receipt'),
                                        'pmcid': record.get('pmcid'), 'final_task_closure': False})))
                db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (task_id,))
        db.commit()
        db.close()


def stage(record: dict, origin: str) -> dict | None:
    """Keep native JATS semantics; DOI mismatch never becomes qualification evidence."""
    if record.get('source') != 'pmc_front_oai' or record.get('status') != 'parsed':
        return None
    parsed = record['parsed']
    if record['doi'] not in parsed['doi']:
        return None
    return {**record, 'source': 'pmc', 'route': 'pmc_journal_oai',
            'original_parsed_receipt': origin, 'parse_authors_from_raw': True,
            'parsed': {**parsed, 'issns': parsed['issn']}}


def consume(done: set[str]) -> dict:
    jobs = []
    handled = []
    used = set()
    for path in sorted((ROOT / 'fields/qualification/oai_parsed').glob('*.json')):
        origin = str(path.relative_to(ROOT))
        if origin in done:
            continue
        record = json.loads(path.read_text())
        if record.get('source') != 'pmc_front_oai':
            continue
        converted = stage(record, origin)
        if converted and record['doi'] in used:
            continue  # Reconcile additional receipts in the next phase.
        handled.append(origin)
        if not converted:
            continue  # Original failure/identity conflict remains explicit in its receipt.
        target = FOLDER / 'staged' / path.name
        atomic_json(target, converted)
        jobs.append((record['doi'], str(target.relative_to(ROOT))))
        used.add(record['doi'])
    qualification = apply_available(supplied_jobs=jobs) if jobs else {}
    db = connect()
    selected = [(wid, shard, receipt) for doi, receipt in jobs for wid, shard in db.execute(
        "SELECT work_id,shard FROM papers WHERE doi=? AND status='included_verified_research'", (doi,))]
    db.close()
    authors = apply_authors(selected_jobs=selected, workers=8, fill_abstract=True) if selected else {}
    if authors.get('papers_changed'):
        from .official_rule_reuse import run as reuse_entities
        reuse_entities()
    record_attempts(handled)
    done.update(handled)
    atomic_json(FOLDER / 'consumed_receipts.json', sorted(done))
    result = {'handled_receipts': len(handled), 'staged_records': len(jobs),
              'qualification': qualification, 'authors': authors, 'network_requests': 0}
    with (FOLDER / 'consumer_events.jsonl').open('a') as stream:
        stream.write(json.dumps(result) + '\n')
    if jobs:
        refresh()
    return result


def pending_receipts(done: set[str]) -> int:
    """Count actual unconsumed receipts, independent of collector run counters."""
    count = 0
    for path in (ROOT / 'fields/qualification/oai_parsed').glob('*.json'):
        if str(path.relative_to(ROOT)) in done:
            continue
        if json.loads(path.read_text()).get('source') == 'pmc_front_oai':
            count += 1
    return count


def run() -> None:
    with (FOLDER / 'consumer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (FOLDER / 'staged').mkdir(exist_ok=True)
        path = FOLDER / 'consumed_receipts.json'
        done = set(json.loads(path.read_text())) if path.exists() else set()
        last_batch = time.monotonic() - 600
        while True:
            state = json.loads((FOLDER / 'status.json').read_text())
            terminal = state['status'] in {'source_stopped', 'batch_paths_complete_remaining_requires_other_paths', 'batch_pass_finished_deferred_sets_retained'}
            waiting = pending_receipts(done)
            age = time.monotonic() - last_batch
            if waiting > 0 and (waiting >= 500 or terminal or age >= 600):
                consume(done)
                last_batch = time.monotonic()
                waiting = pending_receipts(done)
            atomic_json(FOLDER / 'consumer_status.json', {
                'status': 'terminal_cache_consumed' if terminal else 'waiting_for_increment',
                'consumed': len(done), 'available': len(done) + waiting,
                'pending_receipts': waiting, 'collector_run_records': state['new_front_records'],
                'network_requests': 0, 'qualification_completion_not_implied': True,
                'max_batch_wait_seconds': 600})
            if terminal and waiting == 0:
                return
            time.sleep(60)


if __name__ == '__main__':
    run()

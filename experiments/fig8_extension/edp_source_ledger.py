"""Record each EDP cache review outcome without changing qualification or closing alternatives."""
from __future__ import annotations

import fcntl
import json
from collections import Counter
from datetime import datetime, timezone

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> dict:
    folder = ROOT / 'supplement/edp_oai'
    review = json.loads((folder / 'cache_review.json').read_text())
    plan = json.loads((folder / 'plan.json').read_text())
    exact = {r['doi']:r for r in json.loads((folder / 'exact_tail_attempts.json').read_text())}
    outcomes = {r['candidate']['doi']:{'doi':r['candidate']['doi'], 'status':r['status'], 'receipt':r['raw_receipt'],
                'review_receipt':'supplement/edp_oai/cache_review.json', 'set':r['set']} for r in review['records']}
    unreturned = []
    for group in plan['sets']:
        progress = json.loads((folder / (group['set'] + '_progress.json')).read_text())
        for paper in group['targets']:
            doi = paper['doi']
            if doi in outcomes:
                continue
            result = exact.get(doi)
            entry = {**paper,'set':group['set'], 'status':result['status'] if result else 'not_returned_no_exact_response',
                     'receipt':result['receipt']['raw_receipt'] if result else 'supplement/edp_oai/'+group['set']+'_progress.json',
                     'errors':result.get('errors',[]) if result else [], 'review_receipt':'supplement/edp_oai/exact_tail_attempts.json'}
            outcomes[doi] = entry
            unreturned.append(entry)
    mapped_count = len(outcomes)
    extra_path = folder / 'unmapped_record_attempts.json'
    extra = json.loads(extra_path.read_text()) if extra_path.exists() else {}
    unmatched_dois = {p['doi'] for p in plan['unmatched_candidates']}
    for doi, result in extra.items():
        if doi not in unmatched_dois:
            raise ValueError('EDP unmatched retrieval is outside the fixed scope')
        outcomes[doi] = {'doi': doi, 'status': result['status'],
            'receipt': result['receipt'].get('raw_receipt', 'supplement/edp_oai/unmapped_record_attempts.json'),
            'review_receipt': 'supplement/edp_oai/unmapped_record_attempts.json',
            'journal_mapping_confirmed': False, 'native_sets': result.get('native_sets', []),
            'errors': result.get('errors', []), 'identity_review_complete': False}
    source = 'edp_official_oai_cache_review'
    counts = Counter()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        db = connect()
        for doi, item in outcomes.items():
            for tid, in db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?",(doi,)).fetchall():
                if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',(tid,source,item['receipt'])).fetchone():
                    counts['already_recorded'] += 1
                    continue
                detail = {**item,'date_semantics_unconfirmed':True,'generic_dc_type_not_research':True,
                          'final_task_closure':False,'network_requests_this_ledger_step':0}
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                           (tid,source,item['status'],item['receipt'],json.dumps(detail,ensure_ascii=False)))
                db.execute('UPDATE tasks SET attempt_n=attempt_n+1,updated_at=CURRENT_TIMESTAMP WHERE task_id=?',(tid,))
                counts['source_review_events_added'] += 1
        db.commit()
        db.close()
    summary = {'at':datetime.now(timezone.utc).isoformat(),'scope':'Fixed existing EDP candidate scope; mapped sets and unmatched identifier followup reported separately',
               'target_dois':mapped_count + len(unmatched_dois),
               'mapped_target_dois':mapped_count,'mapped_cached_dois':len(review['records']),'unreturned_candidates':unreturned,
               'unmatched_identifier_outcomes':dict(Counter(r['status'] for r in extra.values())),
               'unmatched_identifier_unattempted':len(unmatched_dois - set(extra)),
               'unmatched_journal_candidates':plan['unmatched_candidates'],'review_summary':review['summary'],
               'batch_sequences_complete':True,'exact_omission_followup':'complete','mapped_acquisition_complete':True,
               'acquisition_complete':unmatched_dois <= set(extra) and all(r['status'] != 'source_request_failed' for r in extra.values()),
               'qualification_complete':False,'final_task_closure':False,
               'limitation':'Missing identifiers, generic types/dates and incomplete source titles remain unresolved. Unmatched-journal native identifiers require identity and journal review; other applicable sources are not declared exhausted.'}
    atomic_json(folder / 'completeness.json',summary)
    result = {**counts,'scope_dois':len(outcomes),'qualification_changes':0,'network_requests':0}
    atomic_json(folder / 'ledger_status.json',result)
    print(json.dumps(result),flush=True)
    return result


if __name__ == '__main__':
    run()

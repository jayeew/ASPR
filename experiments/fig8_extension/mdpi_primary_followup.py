"""Finite missing-field archive follow-up after native MDPI type confirmation."""
from __future__ import annotations

import fcntl
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .derived_refresh import refresh
from .publisher_delta_apply import apply_available
from .springer_author_cache import run as apply_authors
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, request


def extend_scope() -> int:
    """Append only newly type-confirmed date gaps with an unreviewed archive lead."""
    path = ROOT / 'supplement/mdpi_oai/primary_followup/scope.json'
    scope = json.loads(path.read_text())
    known_dois = {p['doi'] for p in scope['papers']}
    known_pmcids = {p['pmcid'] for p in scope['papers']}
    db = connect()
    candidates = db.execute("SELECT work_id,doi,pmcids FROM papers WHERE doi LIKE '10.3390/%' "
                            "AND status='pending_primary_online_date'").fetchall()
    archive_dois: dict[str, set[str]] = {}
    for _, doi, value in candidates:
        for pmcid in set(json.loads(value or '[]')):
            archive_dois.setdefault(pmcid, set()).add(doi)
    added = []
    for wid, doi, value in candidates:
        pmcids = sorted(set(json.loads(value or '[]')))
        if doi in known_dois or len(pmcids) != 1 or pmcids[0] in known_pmcids:
            continue
        if len(archive_dois[pmcids[0]]) != 1:
            continue
        evidence = [json.loads(r[0]) for r in db.execute(
            "SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (wid,))]
        if any(e.get('family') == 'archived_jats' and e.get('identity') == 'matched' for e in evidence):
            continue  # Existing archive date gaps need a different source, not another fetch.
        added.append({'work_id': wid, 'doi': doi, 'pmcid': pmcids[0], 'route': 'mdpi'})
        known_dois.add(doi)
        known_pmcids.add(pmcids[0])
    db.close()
    if added:
        scope['papers'].extend(added)
        atomic_json(path, scope)
        with (path.parent / 'scope_events.jsonl').open('a') as stream:
            stream.write(json.dumps({'reason': 'Automatic incremental native-type/date followup',
                                     'new_papers': added, 'network_requests': 0}) + '\n')
    return len(added)


def record_attempts(results: dict) -> dict:
    """Record cached acquisition outcomes once without asserting qualification."""
    counts = {'added': 0, 'already_recorded': 0, 'missing_task': 0}
    source = 'mdpi_missing_date_epmc_front'
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        for pmcid, record in results.items():
            native_receipt = record.get('parsed_receipt') or record.get('raw_receipt')
            receipt = ('fields/qualification/' + native_receipt if native_receipt else
                       'supplement/mdpi_oai/primary_followup/results.json')
            assert (ROOT / receipt).exists(), receipt
            tasks = db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?",
                               (record['doi'],)).fetchall()
            counts['missing_task'] += not bool(tasks)
            for tid, in tasks:
                if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                              (tid, source, receipt)).fetchone():
                    counts['already_recorded'] += 1
                    continue
                detail = {'pmcid': pmcid, 'http_status': record.get('http_status'),
                          'parsed_does_not_imply_qualified': True,
                          'scope': 'supplement/mdpi_oai/primary_followup/scope.json'}
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                           (tid, source, record['status'], receipt, json.dumps(detail)))
                db.execute('UPDATE tasks SET attempt_n=attempt_n+1,updated_at=CURRENT_TIMESTAMP WHERE task_id=?', (tid,))
                counts['added'] += 1
        db.commit()
        db.close()
    atomic_json(ROOT / 'supplement/mdpi_oai/primary_followup/ledger_status.json',
                {**counts, 'scope_responses': len(results), 'eligibility_not_overridden': True})
    return counts


def run() -> None:
    folder = ROOT / 'supplement/mdpi_oai/primary_followup'
    with (folder / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        extend_scope()
        papers = json.loads((folder / 'scope.json').read_text())['papers']
        path = folder / 'results.json'
        results = json.loads(path.read_text()) if path.exists() else {}
        stopped = any(r.get('http_status') in {401, 403, 429} for r in results.values())
        pending = iter([p for p in papers if p['pmcid'] not in results] if not stopped else [])
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = {}
            def submit() -> None:
                paper = next(pending, None)
                if paper:
                    jobs[pool.submit(request, paper, ROOT, 'pmc')] = paper
            for _ in range(8):
                submit()
            while jobs:
                completed, _ = wait(jobs, return_when=FIRST_COMPLETED)
                for future in completed:
                    paper = jobs.pop(future)
                    record = future.result()
                    results[paper['pmcid']] = record
                    stopped |= record.get('http_status') in {401, 403, 429}
                    atomic_json(path, results)
                    if not stopped:
                        submit()
        record_attempts(results)
        jobs = [(r['doi'], 'fields/qualification/' + r['parsed_receipt']) for r in results.values() if r['status'] == 'parsed']
        result = apply_available(supplied_jobs=jobs) if jobs else {}
        from .mdpi_pubmed_increment import run as followup_pubmed
        pubmed = followup_pubmed(results)
        db = connect()
        selected = [(wid, shard, receipt) for doi, receipt in jobs for wid, shard in db.execute(
            "SELECT work_id,shard FROM papers WHERE doi=? AND status='included_verified_research'", (doi,))]
        db.close()
        authors = apply_authors(selected_jobs=selected, workers=8, fill_abstract=True) if selected else {}
        if authors.get('papers_changed'):
            from .official_rule_reuse import run as reuse_entities
            reuse_entities()
        atomic_json(folder / 'status.json', {'scope': len(papers), 'responses': len(results),
                    'parsed': len(jobs), 'source_stopped': stopped, 'qualification': result, 'authors': authors,
                    'pubmed_followup': pubmed,
                    'unattempted': [p['pmcid'] for p in papers if p['pmcid'] not in results],
                    'field_completeness_not_implied': True})
        refresh()
        print(json.dumps({'qualification': result, 'authors': authors}), flush=True)


if __name__ == '__main__':
    run()

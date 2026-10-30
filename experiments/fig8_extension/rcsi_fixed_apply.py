"""Reparse cached RCSI metadata and apply once after the fixed collector finishes."""
from __future__ import annotations

import fcntl
import gzip
import json
from pathlib import Path


def run() -> None:
    folder = Path('outputs/fig8_data/supplement/rcsi_source_review')
    with (folder/'finisher.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        waiting = folder/'application_status.tmp'
        waiting.write_text(json.dumps({'status':'waiting_for_fixed_collector','final_task_closure':False}))
        waiting.replace(folder/'application_status.json')
        with (folder/'collector.lock').open('a') as collector:
            fcntl.flock(collector,fcntl.LOCK_EX)
            # Load current qualification rules after acquisition finishes, not before a long wait.
            from .derived_refresh import refresh
            from .official_rule_reuse import run as reuse_rules
            from .publisher_delta_apply import apply_available
            from .springer_author_cache import run as apply_authors
            from .supplement_queue import ROOT, connect
            from .supplement_sources import atomic_json, jats_metadata
            jobs = []; repaired = 0
            with (folder/'parse_changes.jsonl').open('a') as changes:
                for path in (folder/'records').glob('*.json'):
                    record = json.loads(path.read_text())
                    if record.get('identity_matched') is not True:
                        continue
                    raw = gzip.decompress((ROOT/'fields/qualification'/record['raw_receipt']).read_bytes())
                    parsed = jats_metadata(raw);parsed['doi'] = [d.lower() for d in parsed['doi']];parsed['issns'] = parsed['issn']
                    if record['doi'] not in parsed['doi']:
                        raise ValueError('Cached native DOI changed during reparse')
                    if parsed != record['parsed']:
                        changes.write(json.dumps({'doi':record['doi'],'receipt':str(path.relative_to(ROOT)),
                            'before':record['parsed'],'after':parsed,
                            'reason':'Case-insensitive XML language tags; original English abstract retained, translations excluded'})+'\n')
                        record['parsed'] = parsed;atomic_json(path,record);repaired += 1
                    jobs.append((record['doi'],str(path.relative_to(ROOT))))
            atomic_json(folder/'application_status.json',{'status':'applying_cached_evidence','records':len(jobs),'reparsed_records_changed':repaired,'network_requests':0})
            result = apply_available(supplied_jobs=sorted(set(jobs)))
            atomic_json(folder/'qualification_application.json',result)
            db = connect()
            selected = [(wid,shard,receipt) for doi,receipt in jobs for wid,shard in db.execute(
                "SELECT work_id,shard FROM papers WHERE doi=? AND status='included_verified_research'",(doi,))]
            db.close()
            authors = apply_authors(selected_jobs=selected,workers=8,fill_abstract=True) if selected else {}
            if authors.get('papers_changed'):
                reuse_rules()
            refresh()
            atomic_json(folder/'application_status.json',{'status':'available_fixed_source_records_applied',
                'qualification':result,'authors':authors,'reparsed_records_changed':repaired,
                'network_requests':0,'final_task_closure':False,'full_refresh':'scheduled_only'})
            print(json.dumps({'qualification':result,'authors':authors}),flush=True)


if __name__ == '__main__':
    run()

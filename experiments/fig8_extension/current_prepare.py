"""Execute or finish the approved bounded Fig8 preparation in the current data directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

from .current_extract import OUT as CLAIMS, PROTOCOL, SUCCESS, export_claims, run as extract, write_json
from .current_graph import run as graph
from .current_metadata import run as metadata
from .supplement_queue import ROOT

ANALYSIS=ROOT/'analysis'


def measured_budget() -> None:
    path=CLAIMS/'measured_budget.json'
    if path.exists():
        return
    db=sqlite3.connect(CLAIMS/'contributions.sqlite')
    rows=list(db.execute('''SELECT elapsed,response_json,created_at FROM attempts
        WHERE protocol_hash=? AND status IN ('complete','zero_contributions') ORDER BY id LIMIT 200''',(PROTOCOL,)))
    db.close()
    if len(rows)<200:
        return
    elapsed=(datetime.fromisoformat(rows[-1][2])-datetime.fromisoformat(rows[0][2])).total_seconds()
    throughput=(len(rows)-1)/elapsed*3600 if elapsed else None
    usage=[json.loads(r[1]).get('usage',{}) for r in rows]
    means={k:sum(u.get(k,0) for u in usage)/len(rows) for k in
        ['input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens']}
    estimate={k:round(v*3344) for k,v in means.items()}
    write_json(path,{'successful_papers_measured':len(rows),'observed_papers_per_hour':throughput,
        'mean_reported_usage_per_request':means,'projected_3344_request_usage':estimate,
        'remaining_hours_at_observed_rate':(3344-len(rows))/throughput if throughput else None,
        'note':'Includes CLI system context, which exceeds abstract-only prompt estimates; quota consumption is not a dollar quote.'})


def finish() -> dict[str,Any]:
    db=sqlite3.connect(CLAIMS/'contributions.sqlite')
    available=export_claims(db);db.close()
    measured_budget()
    graph_summary=graph(available)
    meta=json.loads((ANALYSIS/'tables/metadata_summary.json').read_text())
    db=sqlite3.connect(CLAIMS/'contributions.sqlite')
    attempts=db.execute('SELECT count(*) FROM attempts WHERE protocol_hash=?',(PROTOCOL,)).fetchone()[0]
    targets=db.execute('SELECT count(*) FROM results WHERE protocol_hash=? AND status IN (?,?)',
                      (PROTOCOL,*sorted(SUCCESS))).fetchone()[0]
    unresolved=[dict(doi=r[0],work_id=r[1],journal=r[2],status=r[3] or 'pending',
        attempts=r[4],reason=r[5] or 'Not yet requested or provider stopped dispatch') for r in db.execute('''
        SELECT p.doi,p.work_id,p.journal,r.status,p.attempts,r.error FROM papers p
        LEFT JOIN results r USING(doi) WHERE p.source_id IN
        ('https://openalex.org/S137773608','https://openalex.org/S3880285',
         'https://openalex.org/S110447773','https://openalex.org/S125754415')
        AND p.doi IN (SELECT doi FROM cohort_selection WHERE included=1)
        AND (r.status IS NULL OR r.status NOT IN ('complete','zero_contributions'))''')]
    db.close()
    from .current_metadata import write_csv
    if unresolved:
        write_csv(CLAIMS/'unresolved.csv',unresolved)
    else:
        (CLAIMS/'unresolved.csv').write_text('doi,work_id,journal,status,attempts,reason\n')
    ready={r['journal']:r['extracted']/r['papers']>=.95 for r in graph_summary['journal_status']}
    computable={r['journal']:r['computable'] for r in graph_summary['journal_status']}
    result={'updated_at':datetime.now(timezone.utc).isoformat(),
        'status':'complete' if not unresolved else 'complete_with_unresolved_extractions',
        'new_acquisition':False,'old_contributions_used':False,'model':'gpt-6.1-sol','reasoning_effort':'low',
        'extraction_target':3344,'extraction_complete':targets,'extraction_unresolved':len(unresolved),
        'requests_used':attempts,'request_limit':3678,'journal_extraction_gate_95pct':ready,
        'panel_a':{'papers':1595,'data':'tables/panel_a_papers.csv',
            'status':'data_ready' if ready['Nature'] and computable['Nature']>=30 else 'preview_only'},
        'panel_b':{'metadata_papers':419081,'journal_field_points':meta['display_journal_field_points'],
            'metadata_status':'data_ready','graph_status':'data_ready' if all(ready.values()) and min(computable.values())>=30 else 'limited',
            'data':['tables/panel_b_journals.csv','tables/panel_b_four_journal_papers.csv','tables/panel_b_graph_distributions.csv']},
        'panel_c':{'country_papers':meta['country_papers'],'institution_papers':meta['institution_papers'],
            'status':'ready','data':['tables/panel_c_countries.csv','tables/panel_c_institutions.csv']},
        'graph':graph_summary,'metadata':meta,
        'comparison_scope':'Observed screened sample, not full publication totals; graph describes a Nature-Communications-heavy fixed reference.',
        'institution_names_missing':meta['institution_names_missing'],
        'institution_names_missing_note':'Current IDs without display names remain separate and are not reallocated.',
        'extra_llm_analysis_calls':0}
    write_json(ANALYSIS/'preparation_summary.json',result)
    print(json.dumps(result),flush=True)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--refresh-metadata',action='store_true')
    parser.add_argument('--finish-only',action='store_true')
    args=parser.parse_args()
    with (ANALYSIS/'preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.refresh_metadata or not (ANALYSIS/'tables/metadata_summary.json').exists():
            metadata()
        if args.wait_pid:
            deadline=time.monotonic()+12*3600
            while time.monotonic()<deadline:
                try:
                    os.kill(args.wait_pid,0)
                except ProcessLookupError:
                    break
                measured_budget()
                time.sleep(10)
            else:
                write_json(ANALYSIS/'preparation_summary.json',{'status':'waiting_boundary_reached',
                    'reason':'Extraction process has not exited; no competing graph export started.'})
                return
        elif not args.finish_only:
            with (CLAIMS/'worker.lock').open('a') as worker:
                fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
                extract()
        finish()


if __name__=='__main__':
    main()

"""Reconcile a completed fixed PMC batch under current rules and summarize its outcomes."""
from __future__ import annotations

import fcntl
import json
from collections import Counter

from .separate_concern_notice import run as reconcile_notices
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder=ROOT/'supplement/secondary_link_pmc';cloud=ROOT/'supplement/pmc_cloud'
    with (folder/'finisher.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (folder/'pipeline.lock').open('a') as pipeline:
            fcntl.flock(pipeline,fcntl.LOCK_EX)
            status=json.loads((folder/'status.json').read_text())
            if status['status']!='fixed_metadata_and_available_xml_processed':
                raise RuntimeError('Fixed pipeline not successfully terminal; retain unresolved paths')
            reconcile_notices()
            scope=json.loads((folder/'scope.json').read_text())['jobs'];versions=Counter();xml=Counter();assessments=Counter()
            papers={paper['work_id']:paper for group in scope.values() for paper in group}
            for pmc in scope:
                version=cloud/'version_results'/(pmc+'.1.json')
                result=cloud/'xml_results'/(pmc+'.1.json')
                if version.exists():
                    data=json.loads(version.read_text());versions[data['status']]+=1
                    assessments.update(a['status'] for a in data.get('assessments',[]))
                else:versions['missing_result']+=1
                if result.exists():xml[json.loads(result.read_text())['status']]+=1
                else:xml['missing_result']+=1
            db=connect();qualification=Counter();by_journal={}
            for wid,paper in papers.items():
                row=db.execute('SELECT status FROM papers WHERE work_id=?',(wid,)).fetchone()
                qualification[row[0] if row else 'missing_work']+=1
                by_journal.setdefault(paper['journal'],Counter())[row[0] if row else 'missing_work']+=1
            db.close()
            atomic_json(folder/'summary.json',{'status':'fixed_batch_processed_not_global_completion',
                'identifiers':len(scope),'distinct_work_ids':len(papers),'metadata_outcomes':dict(versions),
                'assessment_outcomes':dict(assessments),'xml_outcomes':dict(xml),
                'current_qualification':dict(qualification),'by_journal':by_journal,
                'notice_rule_reconciliation':'supplement/separate_concern_notices/application.json',
                'network_requests_in_finisher':0,'global_coverage_refresh':'pending_combined_refresh',
                'final_task_closure':False})
            print(json.dumps({'fixed_work_ids':len(papers),'qualification':dict(qualification),'xml':dict(xml)}),flush=True)


if __name__ == '__main__':
    run()

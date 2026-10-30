from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from experiments.fig8_extension.metadata_gap_flags import run
from experiments.fig8_extension.supplement_queue import connect


def test_gap_cache_matches_recomputed_roster_and_repeat_is_stable(tmp_path: Path) -> None:
    db=connect(tmp_path)
    db.executemany('INSERT INTO papers(work_id,status,abstract_gap,institution_gap,country_gap) VALUES(?,?,?,?,?)',
        [('W1','included_verified_research',1,1,1),('W2','included_verified_research',0,0,0),('W3','excluded_primary_nonresearch',1,1,1)])
    db.commit();db.close()
    folder=tmp_path/'fields/qualification';folder.mkdir(parents=True)
    (folder/'research_metadata_gaps_summary.json').write_text(json.dumps({'qualified_denominator':2,'any_metadata_gap':1,'missing_abstract':0,'missing_institution':0,'missing_country':1}))
    with gzip.open(folder/'research_metadata_gaps.csv.gz','wt') as stream:
        writer=csv.DictWriter(stream,fieldnames=['work_id','missing_abstract','missing_institution','missing_country'])
        writer.writeheader();writer.writerow({'work_id':'W2','missing_abstract':False,'missing_institution':False,'missing_country':True})
    result=run(tmp_path)
    assert result['after']=={'abstract':0,'institution':0,'country':1}
    assert run(tmp_path)['before']==result['after']
    db=connect(tmp_path)
    assert db.execute('SELECT work_id,abstract_gap,institution_gap,country_gap FROM papers ORDER BY work_id').fetchall()==[('W1',0,0,0),('W2',0,0,1),('W3',1,1,1)]
    db.close()

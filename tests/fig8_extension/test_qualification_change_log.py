import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.fig8_extension.fig8b_qualification_build import write_decisions


def test_field_change_logged_once_and_unchanged_row_preserved(tmp_path):
    path=tmp_path/'decisions.parquet';audit=tmp_path/'changes.jsonl.gz'
    old=[{'work_id':'W1','doi':'10.1/a','evidence_ids':'[]','status':'pending'},
         {'work_id':'W2','doi':'10.1/b','evidence_ids':'[]','status':'pending'}]
    pq.write_table(pa.Table.from_pylist(old),path)
    new=[{**old[0],'status':'included','evidence_ids':'["e1"]'},old[1]]
    write_decisions(path,new,audit)
    write_decisions(path,new,audit)
    with gzip.open(audit,'rt') as f:rows=[json.loads(line) for line in f]
    assert len(rows)==1
    assert rows[0]['changed_fields']['status']=={'before':'pending','after':'included'}
    assert pq.read_table(path).to_pylist()==new

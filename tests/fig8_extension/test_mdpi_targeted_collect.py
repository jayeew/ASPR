from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

from experiments.fig8_extension import mdpi_targeted_collect as mod


def test_exact_identity_stages_once_and_retains_task_status(tmp_path: Path, monkeypatch) -> None:
    folder=tmp_path/'supplement/mdpi_oai';target=folder/'targeted'
    (folder/'type_application').mkdir(parents=True);target.mkdir();(folder/'matched_pages').mkdir()
    (folder/'type_application/applied_fingerprints.json').write_text('{}')
    job={'doi':'10.3390/test1','work_id':'W1','title':'Test title','journal':'Test Journal','source_id':'S1',
         'oai_identifier':'oai:mdpi.com:/1234-5678/2/3/4/','discovery_receipt':'cached-openalex.json'}
    (target/'scope.json').write_text(json.dumps({'papers':[job]}))
    raw=b'''<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/" xmlns:dc="http://purl.org/dc/elements/1.1/"><GetRecord><record><header><identifier>oai:mdpi.com:/1234-5678/2/3/4/</identifier><datestamp>2024-01-01</datestamp><setSpec>research-article</setSpec></header><metadata><dc:title>Test title</dc:title><dc:source>Test Journal</dc:source><dc:identifier>https://doi.org/10.3390/test1</dc:identifier></metadata></record></GetRecord></OAI-PMH>'''
    (tmp_path/'response.gz').write_bytes(gzip.compress(raw))
    database=tmp_path/'tasks.sqlite'
    def connect():
        return sqlite3.connect(database)
    db=connect();db.executescript("CREATE TABLE tasks(task_id TEXT,kind TEXT,entity_key TEXT,status TEXT,attempt_n INTEGER); CREATE TABLE attempts(task_id TEXT,source TEXT,status TEXT,receipt TEXT,detail TEXT); INSERT INTO tasks VALUES('t1','qualification','10.3390/test1','pending',0);");db.close()
    for key,value in [('ROOT',tmp_path),('FOLDER',folder),('TARGET',target),('connect',connect)]:monkeypatch.setattr(mod,key,value)
    monkeypatch.setattr(mod,'read_csv',lambda path:[{'source_id':'S1','issn':'["1234-5678"]'}])
    requests=[]
    def fetch(url,params):
        requests.append(params)
        return {'http_status':200,'raw_receipt':'response.gz','attempts':[{'http_status':200}]}
    monkeypatch.setattr(mod.transport,'fetch',fetch)
    mod.run();mod.run()
    assert len(requests)==1
    state=json.loads((target/'status.json').read_text())
    assert state['matched_records']==1 and state['status']=='fixed_queue_processed'
    assert len(list((folder/'matched_pages').glob('*.json')))==1
    db=connect();assert db.execute('SELECT status,attempt_n FROM tasks').fetchone()==('pending',1)
    assert db.execute('SELECT count(*) FROM attempts').fetchone()==(1,)
    db.close()

"""Resolve two explicit university signatures against native records and official sites."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict

import pyarrow.parquet as pq
from lxml import html

from .official_units_apply import run as apply_rules
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .terminal_country_apply import native_metadata
from .three_journal_apply import read_csv

FOLDER=ROOT/'supplement/official_entities/ufjf_uta_units'
DEFINITIONS=[('ufjf','https://openalex.org/I101100930','Universidade Federal de Juiz de Fora',
    'departamento de engenharia elétrica, universidade federal de juiz de fora (ufjf), juiz de fora, brazil'),
    ('uta','https://openalex.org/I185652977','University of Tarapacá',
    'instituto de alta investigación, universidad de tarapacá, arica, chile')]


def prepare_rules() -> list[dict]:
    rules=[]
    for label,eid,name,signature in DEFINITIONS:
        source=json.loads((FOLDER/f'{label}.json').read_text())
        raw=gzip.decompress((ROOT/source['raw_receipt']).read_bytes())
        assert source['status']==200 and hashlib.sha256(raw).hexdigest()==source['sha256']
        text=' '.join(html.fromstring(raw.decode('utf-8')).text_content().split())
        phrases=(['Universidade Federal de Juiz de Fora','A UFJF oferece cinco cursos de graduação em Engenharia Elétrica'] if label=='ufjf' else ['Universidad de Tarapacá','Instituto de Alta Investigación'])
        assert all(p in text for p in phrases)
        rules.append(dict(entity_id=eid,name=name,exact_names=[signature],entity_type='education',
            ownership_status='officially_confirmed_university_self',university_targets={eid:1.0},
            source=str((FOLDER/f'{label}.json').relative_to(ROOT)),official_url=source['url'],
            raw_receipt=source['raw_receipt'],raw_sha256=source['sha256'],
            scope='Exact full publisher signature explicitly names this university and locality; contemporary article establishes affiliation. Official university page corroborates university identity and academic activity. Map the named university itself; UFJF department naming is not normalized or equated with any other department. No hospital, branch, joint-center or country inference.'))
    return rules


def run() -> None:
    if (FOLDER/'result.json').exists():
        print('Completed fixed signature queue; no unchanged-source rerun');return
    rules=prepare_rules();aliases={n for r in rules for n in r['exact_names']}
    db=connect();groups=defaultdict(set)
    for alias in aliases:
        for wid,shard in db.execute("SELECT DISTINCT p.work_id,p.shard FROM tasks t JOIN task_papers l USING(task_id) JOIN papers p USING(work_id) WHERE t.kind='institution' AND t.entity_key=? AND p.status='included_verified_research'",(alias,)):
            groups[shard].add(wid)
    db.close();journals={r['source_id']:set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT/'fields/journal_registry.csv')}
    selected=set();checks=[]
    for shard,ids in groups.items():
        for paper in pq.read_table(ROOT/'fields/papers'/shard,filters=[('work_id','in',sorted(ids))]).to_pylist():
            receipts=defaultdict(set)
            for author in json.loads(paper['authors_json']):
                if author.get('nature_v1_slots') is not None or not author.get('publisher_affiliation_source'):continue
                for slot in author.get('publisher_affiliation_slots') or []:
                    name=' '.join(slot.get('text','').casefold().split())
                    if name in aliases:receipts[author['publisher_affiliation_source']].add(name)
            matched=set()
            for receipt,names in receipts.items():
                native=native_metadata(receipt,paper['doi'])
                native_names={' '.join(s['text'].casefold().split()) for a in native.get('authors',[]) for s in a.get('affiliations',[])}
                valid=paper['doi'] in native.get('doi',[]) and signature_match(paper,native,journals[paper['source_id']])
                found=names&native_names if valid else set();matched.update(found)
                checks.append(dict(work_id=paper['work_id'],doi=paper['doi'],receipt=receipt,expected_names=sorted(names),confirmed_names=sorted(found)))
            if receipts and set.union(*receipts.values())<=matched:selected.add(paper['work_id'])
    atomic_json(FOLDER/'rules.json',dict(status='reviewed',rules=rules,native_checks=checks,final_task_closure=False))
    atomic_json(FOLDER/'scope.json',dict(candidate_papers=sum(map(len,groups.values())),native_confirmed_papers=len(selected),selected_work_ids=sorted(selected),held_work_ids=sorted(set.union(*groups.values())-selected),network_requests_for_native_checks=0))
    result=apply_rules(rules_path=str((FOLDER/'rules.json').relative_to(ROOT)),status_name=f'{FOLDER.name}/application.json',selected_work_ids=selected)
    atomic_json(FOLDER/'result.json',dict(application=result,native_confirmed_papers=len(selected),paper_network_requests=0,final_task_closure=False))
    print(json.dumps(result),flush=True)


if __name__=='__main__':run()

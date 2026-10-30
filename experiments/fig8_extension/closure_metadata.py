"""Cache-only abstract and signature-note review for the finite closure goal."""
from __future__ import annotations

import csv
import gzip
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

from .closure_goal import AUDIT
from .three_journal_apply import target_papers, write_csv
from .three_journal_round import OUT, Q, ROOT, load, save


def norm(value: str) -> str:
    return ' '.join(value.casefold().split())


def repair_abstract() -> None:
    marker=AUDIT/'abstract_repair.json'
    if marker.exists():
        return
    receipt='fields/qualification/oai_raw/PMC11623344.xml.gz'
    tree=etree.fromstring(gzip.decompress((ROOT/receipt).read_bytes()),etree.XMLParser(resolve_entities=False,no_network=True))
    article=tree.xpath('//*[local-name()="metadata"]/*[local-name()="article"]')[0]
    dois=article.xpath('./*[local-name()="front"]/*[local-name()="article-meta"]/*[local-name()="article-id"][@pub-id-type="doi"]/text()')
    assert dois==['10.1016/j.cell.2024.08.029']
    abstracts=article.xpath('./*[local-name()="front"]/*[local-name()="article-meta"]/*[local-name()="abstract"][not(@abstract-type)]')
    assert len(abstracts)==1
    text=' '.join(' '.join(n.itertext()).strip() for n in abstracts[0].xpath('./*[local-name()="p"]'))
    assert len(text.split())>100
    db=sqlite3.connect(ROOT/'supplement/tasks.sqlite')
    row=db.execute('SELECT work_id,shard FROM papers WHERE doi=?',(dois[0],)).fetchone()
    path=ROOT/'fields/papers'/row[1]
    table=pq.read_table(path);rows=table.to_pylist();before=None
    for p in rows:
        if p['work_id']==row[0]:
            assert norm(p['abstract']) in norm(text), 'Non-substring conflict requires review'
            before={'abstract':p['abstract'],'abstract_source':p['abstract_source']}
            p['abstract']=text;p['abstract_source']='archived_publisher_primary_abstract'
            for field,old in before.items():
                db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                           (p['work_id'],field,json.dumps(old),json.dumps(p[field]),receipt+'#primary-abstract','Cached primary Summary repairs a verified truncated substring; graphical/In Brief excluded'))
    assert before is not None
    temp=path.with_suffix('.parquet.tmp');pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),temp);temp.replace(path)
    db.commit();db.close();save(marker,{'work_id':row[0],'doi':dois[0],'before':before,'after':text,'receipt':receipt+'#primary-abstract','changed_rows':1,'abstract_nonempty_count_delta':0})


def run() -> None:
    repair_abstract()
    D,_,_,_=load();P=target_papers();selection=json.loads((AUDIT/'institution_top30.json').read_text())
    selected={r['entity_key']:r for r in selection};evidence=defaultdict(list)
    for wid,d in D.items():
        if d['eligibility_status']!='included_verified_research':continue
        p=P[wid]
        for author in json.loads(p['authors_json']):
            for slot in author.get('publisher_affiliation_slots') or []:
                n=norm(slot['text'])
                if n in selected and len(evidence[n])<3:
                    source=author.get('publisher_affiliation_source','')
                    if source and (ROOT/source).is_file():
                        evidence[n].append({'work_id':wid,'doi':d['doi'],'text':slot['text'],'source':source,'author_position':author['position']})
    reviewed=[];notes=set()
    for n,r in selected.items():
        note=bool(re.fullmatch(r'\d+\s*(lead contact|these authors contributed equally)',n)) and bool(evidence[n])
        if note:notes.add(n)
        reviewed.append({**r,'status':'not_an_institution' if note else 'unresolved_official_identity_evidence_missing',
                         'reason':'Author contact/contribution note; no institution or country assigned' if note else 'Cached entity registry is inherited OpenAlex/ROR, not independent official signature/ownership proof; no assignment made',
                         'evidence':evidence[n]})
    save(AUDIT/'institution_cache_review.json',reviewed)
    save(AUDIT/'noninstitution_note_rules.json',{'exact_normalized_texts':sorted(notes),'source':'institution_cache_review.json','scope':'Exclude these publisher contact/contribution notes from geographic institution-slot denominator; retain raw signature text'})
    marker=AUDIT/'institution_ledger_applied.json'
    if not marker.exists():
        db=sqlite3.connect(ROOT/'supplement/tasks.sqlite')
        for r in reviewed:
            state='not_an_institution' if r['status']=='not_an_institution' else 'pending'
            db.execute('UPDATE tasks SET status=?,reason=?,receipt=?,updated_at=CURRENT_TIMESTAMP WHERE task_id=?',
                       (state,r['reason'],'supplement/closure_goal/institution_cache_review.json',r['task_id']))
            db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                       (r['task_id'],'cached_signature_entity_review',r['status'],'supplement/closure_goal/institution_cache_review.json',json.dumps({'reason':r['reason'],'network_requests':0})))
        db.commit();db.close();save(marker,{'reviewed':len(reviewed),'noninstitution_note_entities':len(notes),'confirmed_new_institution_or_country_assignments':0})
    aggregate=defaultdict(Counter);provenance=defaultdict(Counter)
    for wid,d in D.items():
        if d['eligibility_status']!='included_verified_research':continue
        p=P[wid];c=aggregate[p['journal_name']];authors=json.loads(p['authors_json']);c['papers']+=1;c['authors']+=len(authors)
        c['papers_all_authors_known_institution']+=bool(authors) and all(a['institution_ids_or_verified_local_entities'] for a in authors)
        c['papers_all_authors_known_country']+=bool(authors) and all(a['countries'] for a in authors)
        c['papers_author_list_may_be_truncated']+=bool(p['author_list_may_be_truncated'])
        for author in authors:
            c['authors_known_institution']+=bool(author['institution_ids_or_verified_local_entities']);c['authors_known_country']+=bool(author['countries'])
            c['authors_country_conflict']+=bool(author.get('country_conflict_retained'))
            provenance[p['journal_name']][author.get('identity_source') or 'unspecified']+=1
            slots=author.get('publisher_affiliation_slots') or []
            c['authors_without_publisher_slot_structure']+=not slots
            for slot in slots:
                if norm(slot['text']) in notes:
                    c['noninstitution_note_slots_excluded']+=1;continue
                c['represented_publisher_institution_slots']+=1
                c['slots_known_institution']+=bool(slot.get('institution_ids'))
                c['slots_known_country']+=bool(slot.get('country_codes'))
                c['slots_unresolved_identity']+=slot.get('identity_status')=='unresolved'
    columns=sorted({k for c in aggregate.values() for k in c})
    write_csv(OUT/'three_journal_author_institution_coverage.csv',[{'journal':j,**{k:c[k] for k in columns}} for j,c in aggregate.items()])
    save(AUDIT/'author_provenance.json',dict(provenance))
    print(json.dumps({'entity_reviewed':len(reviewed),'note_entities':len(notes),'journals':dict(aggregate)}))


if __name__=='__main__':
    run()

"""Combine RNA native dated publisher statements with cached explicit online dates."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter

from lxml import etree

from .cell_press_annual_cache import apply_proposals
from .fig8b_qualification_decide import key
from .jats_annual_copyright import named_publisher
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/rna_annual_cache'


def prepare() -> tuple[list[dict], list[dict]]:
    db = connect()
    papers = db.execute("SELECT work_id,doi,title,journal,source_id FROM papers WHERE source_id='https://openalex.org/S187657468'").fetchall()
    rules, reviews, cache = {}, [], {}
    for wid, doi, title, journal, sid in papers:
        primary = [json.loads(p) for p, in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (wid,))]
        dates = [e for e in primary if e.get('identity') == 'matched' and e.get('family') == 'crossref_publisher_deposit' and e.get('online_date', '')[:4] in {'2023','2024','2025'}]
        years = {e['online_date'][:4] for e in dates}
        native = [e for e in primary if e.get('identity') == 'matched' and e.get('family') == 'archived_jats']
        if len(years) != 1:
            reviews.append({'work_id':wid,'status':'no_unique_cached_primary_online_year'}); continue
        date_e = dates[0]; date_path = ROOT/'fields/qualification'/date_e['receipt']
        if str(date_path) not in cache:
            raw_date = gzip.decompress(date_path.read_bytes()); data=json.loads(raw_date)
            message=data['response']['message']; items=message.get('items',[message])
            cache[str(date_path)] = (items, hashlib.sha256(raw_date).hexdigest())
        items,date_hash = cache[str(date_path)]
        matches=[(i,r) for i,r in enumerate(items) if r.get('DOI','').lower()==doi]
        if len(matches)!=1:
            raise ValueError('Cached date identity not unique')
        position, item = matches[0]
        date_values=['-'.join(str(v).zfill(4 if i==0 else 2) for i,v in enumerate(parts)) for parts in item.get('published-online',{}).get('date-parts',[])]
        if date_e['online_date'] not in date_values:
            raise ValueError('Explicit deposited online date not reproduced')
        year=int(next(iter(years)))
        for e in native:
            path=ROOT/'fields/qualification'/e['receipt']; raw=gzip.decompress(path.read_bytes())
            document=raw+b'</article>' if raw.rstrip().endswith(b'</front>') else raw
            tree=etree.fromstring(document,etree.XMLParser(resolve_entities=False,no_network=True))
            native_title=' '.join(tree.xpath('//article-meta/title-group/article-title//text()'))
            native_dois=tree.xpath('//article-meta/article-id[@pub-id-type="doi"]/text()')
            journal_names=tree.xpath('//journal-meta/journal-title-group/journal-title/text()')
            if doi not in [d.lower() for d in native_dois] or key(title)!=key(native_title) or journal not in journal_names:
                raise ValueError('Native article identity changed')
            named=named_publisher(tree,year)
            if named is None:
                reviews.append({'work_id':wid,'receipt':e['receipt'],'status':'no_matching_dated_publisher_statement'});continue
            publisher,statement=named
            if publisher!='Cold Spring Harbor Laboratory Press':raise ValueError('Unexpected publisher')
            proof={'doi':doi,'publication_date':date_e['online_date'],'statement':statement,
                   'raw_receipt':str(path.relative_to(ROOT)),'raw_sha256':hashlib.sha256(raw).hexdigest(),
                   'source_url':e['url'],'locator':'front/article-meta/permissions',
                   'online_date_evidence':{'id':date_e['id'],'receipt':str(date_path.relative_to(ROOT)),
                    'raw_sha256':date_hash,'locator':f'response.message.items[{position}].published-online','url':date_e['url']}}
            pair=(sid,year)
            if pair not in rules:
                rules[pair]={'source_id':sid,'year':year,'label':'False','journal':journal,'publisher':publisher,
                    'source_url':e['url'],'raw_receipt':proof['raw_receipt'],'raw_sha256':proof['raw_sha256'],
                    'first_dated_proof':proof,'last_dated_proof':proof,
                    'scope':'Native named publication statement and copyright year agree with independently identity-matched explicit deposited online year. No cross-year extrapolation; reopen upon transfer evidence.'}
            rule=rules[pair]
            if proof['publication_date']<rule['first_dated_proof']['publication_date']:rule['first_dated_proof']=proof
            if proof['publication_date']>rule['last_dated_proof']['publication_date']:rule['last_dated_proof']=proof
            reviews.append({'work_id':wid,'doi':doi,'year':year,'status':'dated_publisher_confirmed','proof':proof})
    db.close()
    return list(rules.values()),reviews


def run() -> dict:
    FOLDER.mkdir(exist_ok=True)
    proposals,reviews=prepare()
    atomic_json(FOLDER/'review.json',{'records':reviews,'network_requests':0})
    atomic_json(FOLDER/'proposals.json',proposals)
    atomic_json(FOLDER/'discovery_status.json',{'proposed_journal_years':len(proposals),'outcomes':dict(Counter(r['status'] for r in reviews)),'network_requests':0})
    if not proposals:return {'proposals':0}
    atomic_json(FOLDER/'application_status.json',{'status':'waiting_for_writer','proposed_journal_years':len(proposals)})
    result=apply_proposals(proposals,FOLDER,[r for r in reviews if r['status']!='dated_publisher_confirmed'],
        rule_status='dated_native_jats_publication_statement',reason='Matched RNA publication statement and copyright year agree with explicit cached deposited online year',
        route='native_jats_copyright',apply_pairs={(r['source_id'],r['year']) for r in proposals})
    atomic_json(FOLDER/'application_status.json',{'status':'applied','result':result,'network_requests':0,'final_task_closure':False})
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    run()

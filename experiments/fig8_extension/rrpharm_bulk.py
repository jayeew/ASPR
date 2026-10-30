"""Locate fixed transferred-journal DOIs in official OAI metadata, without inferring type."""
from __future__ import annotations

import fcntl
import gzip
import json
import hashlib

from lxml import etree

from . import ojs_metadata_harvest as transport
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/pensoft_bulk/transferred_journals'
URL = 'https://rrpharmacology.ru/index.php/journal/oai'
NS = {'o':'http://www.openarchives.org/OAI/2.0/', 'dc':'http://purl.org/dc/elements/1.1/'}


def run() -> None:
    out = FOLDER / 'rrpharm_matches'
    out.mkdir(exist_ok=True)
    with (FOLDER / 'rrpharm.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transport.FOLDER, transport.RAW = FOLDER, FOLDER / 'raw'
        fixed = set(json.loads((FOLDER.parent/'missing_identifiers/scope.json').read_text())['dois'])
        db = connect()
        targets = {d for d, in db.execute("SELECT doi FROM papers WHERE journal='Research Results in Pharmacology'") if d in fixed}
        db.close()
        state_path = FOLDER / 'rrpharm_checkpoint.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'status':'running','pages':0,'seen':[], 'params':{'verb':'ListRecords','metadataPrefix':'oai_dc','from':'2023-01-01'}}
        if state['status'] != 'running':
            return
        while state['params']:
            receipt = transport.fetch(URL, state['params'])
            if receipt.get('http_status') != 200:
                state.update(status='source_unavailable_remaining_retained',failure_receipt=receipt);break
            tree = etree.fromstring(gzip.decompress((ROOT/receipt['raw_receipt']).read_bytes()),etree.XMLParser(resolve_entities=False,no_network=True))
            for record in tree.findall('o:ListRecords/o:record',NS):
                ids=record.xpath('./o:metadata/*/dc:identifier/text()',namespaces=NS)
                matched={x.lower().strip() for x in ids if x.lower().strip() in targets}
                for doi in matched:
                    atomic_json(out/(hashlib.sha256(doi.encode()).hexdigest()+'.json'),{
                        'doi':doi,'page_receipt':receipt,'identifiers':ids,
                        'titles':record.xpath('./o:metadata/*/dc:title/text()',namespaces=NS),
                        'dates':record.xpath('./o:metadata/*/dc:date/text()',namespaces=NS),
                        'types':record.xpath('./o:metadata/*/dc:type/text()',namespaces=NS),
                        'sources':record.xpath('./o:metadata/*/dc:source/text()',namespaces=NS),
                        'abstracts':record.xpath('./o:metadata/*/dc:description/text()',namespaces=NS),
                        'qualification_not_implied':True})
            token=tree.findtext('o:ListRecords/o:resumptionToken',default='',namespaces=NS).strip()
            state['pages']+=1;state['matched']=len(list(out.glob('*.json')));state['fixed_dois']=len(targets)
            if token and token in state['seen']:
                state['status']='repeated_token_remaining_retained';break
            state['seen'].append(token)
            state['params']={'verb':'ListRecords','resumptionToken':token} if token else {}
            if not token or state['matched']==len(targets):state['status']='fixed_doi_discovery_processed_type_date_review_required'
            atomic_json(state_path,state)
            if state['status']!='running':break
        atomic_json(state_path,state)
        print(json.dumps({k:v for k,v in state.items() if k not in {'seen','params'}}))


if __name__ == '__main__':run()
